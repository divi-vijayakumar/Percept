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
import signal
import site
import socket
import stat
import struct
import subprocess
import sys
import time
import weakref
from collections.abc import AsyncIterator, Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields, replace
from types import FunctionType, MappingProxyType, MethodType
from typing import Any, ClassVar, Protocol, runtime_checkable

from .judge_context_producers import RESERVED_CONTEXT_KEYS
from .adapters import AudioFrame, Verdict
from .config import Settings, load_settings
from .contracts import Box, ClipEvent, SubjectRef

#: The contract version a request gets when it does not name one. ``1.1.0`` adds the gated
#: detection capability (``vision.detect.gated``) and its DTOs; nothing in ``1.0.0`` changed
#: shape or behaviour, so a Cloud deployment pinned to ``1.0.0`` keeps judging unmodified.
HOSTED_BYOK_CONTRACT_VERSION = "1.1.0"
#: Every contract version this SDK still executes. A pinned older version is accepted verbatim and
#: is refused ONLY for capabilities that did not exist in it — never silently upgraded.
HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS: tuple[str, ...] = ("1.0.0", "1.1.0")
#: Capabilities introduced after ``1.0.0``, keyed by the first contract version that carries them.
#: A request pinned below the introducing version is refused at preflight rather than served a
#: surface its caller has not agreed to.
_CAPABILITY_MIN_CONTRACT_VERSION: Mapping[str, str] = MappingProxyType(
    {"vision.detect.gated": "1.1.0"}
)
HOSTED_BYOK_PROCESS_ISOLATION_VERSION = "2.3.0"
HOSTED_BYOK_CAPABILITIES: Mapping[str, frozenset[str]] = {
    "gemini": frozenset({"vision.judge", "vision.judge.fused", "vision.detect.gated"}),
    "openrouter": frozenset({"vision.judge", "vision.detect.gated"}),
}
#: Default per-goal refractory (minimum seconds between two proposals for the same goal). This is
#: the ONLY thing bounding proposal count on a long clip, so it is an explicit contract constant
#: rather than a settings lookup: a hosted caller must be able to compute its own worst case.
HOSTED_BYOK_DEFAULT_REFRACTORY_S = 15.0
#: Detection is defined over TIER-0 GATE FRAMES, not full frames. ``percept.runtime.source``
#: downsamples every captured frame to ``GATE_W x GATE_H`` (48x36 = 1,728 luma bytes) before the
#: gate computes anything, and this boundary refuses anything larger so a full-resolution frame
#: cannot be shipped across the seam by accident. 1,200 gate frames (600 s at 2 fps) is ~2 MB
#: against the 32 MB IPC ceiling.
HOSTED_BYOK_MAX_GATE_FRAME_PIXELS = 48 * 36
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
#: Context keys the RUNTIME owns. A hosted caller supplying one is refused before any adapter is
#: constructed: these select a different request shape, carry state the caller cannot have, or bind a
#: correction channel — none of which a per-invocation lease gets to steer.
#:
#: DERIVED, not restated. This was a hand-maintained list and it had fallen two keys behind the
#: runtime (``event_chain`` and the open-episode key), so a hosted caller could pass both. Deriving it
#: means a key added to the registry is refused here on the same commit.
_RESERVED_CONTEXT_KEYS = RESERVED_CONTEXT_KEYS


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


class HostedByokDetectionStarved(HostedByokError):
    """A gated detection run admitted NOTHING, anywhere, for any goal.

    An empty proposal list is indistinguishable from "nothing happened", and both tier-0 defaults
    this SDK has shipped were at some point silently starving (a whole-frame mean that never rose
    over a still background) or deaf (an audio gate whose onset never fired) while their own tests
    stayed green. A hosted caller cannot see the salience distribution, so a zero-admission run is
    reported as a *diagnostic* carrying the numbers instead of as an empty success.

    ``misconfigured`` is the boundary's read of WHICH failure this is: ``True`` when the run
    produced real salience that never reached the applied cutoff (the gate is deaf to this
    material — retune or supply an admission signal), ``False`` when the observed salience was
    essentially zero (the clip really was still). Both raise; only the caller can decide.
    """

    def __init__(
        self,
        message: str,
        *,
        goal_ids: tuple[str, ...],
        frames: int,
        salience_stat: str,
        thresholds: Mapping[str, float],
        max_salience: float,
        mean_salience: float,
        misconfigured: bool,
    ) -> None:
        super().__init__(message)
        self.goal_ids = goal_ids
        self.frames = frames
        self.salience_stat = salience_stat
        self.thresholds = MappingProxyType(dict(thresholds))
        self.max_salience = max_salience
        self.mean_salience = mean_salience
        self.misconfigured = misconfigured

    def to_dict(self) -> dict[str, Any]:
        """The diagnostic as plain JSON-able scalars, for a hosted error payload."""

        return {
            "code": "hosted_byok_detection_admitted_nothing",
            "goal_ids": list(self.goal_ids),
            "frames": self.frames,
            "salience_stat": self.salience_stat,
            "thresholds": dict(self.thresholds),
            "max_salience": self.max_salience,
            "mean_salience": self.mean_salience,
            "misconfigured": self.misconfigured,
        }


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
        if self.contract_version not in HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS:
            raise HostedByokContractError(
                f"unsupported hosted BYOK contract version {self.contract_version!r}; "
                f"this SDK implements {list(HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS)!r} "
                f"(default {HOSTED_BYOK_CONTRACT_VERSION!r})."
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


# ---------------------------------------------------------------------------
# vision.detect.gated — DTOs
#
# Detection is the ONE hosted route that contacts no provider. It is the tier-0 salience gate
# (``percept.harness.tier0.MotionGate``) run over an ordered sequence of gate-resolution frames;
# it decides WHEN a moment is worth a judgment and returns those moments with temporal identity.
# The judgment itself remains ``judge()``, unchanged.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HostedByokGateFrame:
    """One tier-0 gate frame plus the temporal identity a proposal is bound to.

    ``pixels`` is a flat, row-major tuple of luma values in ``0..255`` of length ``width*height``
    — exactly ``percept.harness.tier0.Frame``'s representation, which is what
    ``percept.runtime.source`` already produces (``GATE_W x GATE_H`` = 48x36) before any gate
    arithmetic happens. Full-resolution frames are deliberately NOT accepted here: they are needed
    only for the moments this route admits, and are then carried by ``judge()``.

    ``frame_index`` is the caller's ordinal within its own capture; ``offset_s`` is seconds from
    the start of the clip and is what the gate's schedule/cadence policy is evaluated against;
    ``timestamp_unix_s`` is the optional absolute wall-clock instant. All three are echoed back on
    every proposal so Cloud can bind an incident to the moment without re-deriving anything.
    """

    width: int
    height: int
    pixels: tuple[int, ...]
    frame_index: int
    offset_s: float
    timestamp_unix_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("width", "height", "frame_index"):
            value = getattr(self, name)
            if type(value) is not int:
                raise HostedByokContractError(f"gate frame {name} must be an exact int.")
        if self.width <= 0 or self.height <= 0:
            raise HostedByokContractError("gate frame dimensions must be positive.")
        if self.frame_index < 0:
            raise HostedByokContractError("gate frame frame_index must not be negative.")
        if self.width * self.height > HOSTED_BYOK_MAX_GATE_FRAME_PIXELS:
            raise HostedByokContractError(
                f"gate frame {self.width}x{self.height} exceeds the tier-0 gate resolution "
                f"({HOSTED_BYOK_MAX_GATE_FRAME_PIXELS} luma samples); downsample to the gate "
                "frame the capture path already produces before crossing this boundary."
            )
        if type(self.pixels) is not tuple:
            raise HostedByokContractError("gate frame pixels must be an exact tuple of ints.")
        if len(self.pixels) != self.width * self.height:
            raise HostedByokContractError(
                f"gate frame pixel count {len(self.pixels)} != {self.width}*{self.height}."
            )
        for pixel in self.pixels:
            if type(pixel) is not int or not 0 <= pixel <= 255:
                raise HostedByokContractError(
                    "gate frame pixels must be exact ints in 0..255 (8-bit luma)."
                )
        for name in ("offset_s", "timestamp_unix_s"):
            value = getattr(self, name)
            if value is None:
                if name == "offset_s":
                    raise HostedByokContractError("gate frame offset_s is required.")
                continue
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise HostedByokContractError(f"gate frame {name} must be a finite number.")
        if float(self.offset_s) < 0:
            raise HostedByokContractError("gate frame offset_s must not be negative.")


@dataclass(frozen=True, slots=True)
class HostedByokDetectionGoal:
    """One goal to gate for, in the exact shape a ``WatchCondition`` needs.

    This boundary deliberately does NOT interpret ``condition``: the text rides through as the
    condition's ``anchor_text`` for provenance only. Choosing or rewriting goal semantics is not
    this seam's job and never becomes one.

    ``threshold`` / ``salience_stat`` default to the deployment's tier-0 configuration, resolved
    with the SAME per-stat rule ``percept.harness.tier0.compile_watchspec`` applies — the
    localized ``block_max`` statistic sits on a higher value scale than a whole-frame ``mean``
    and gets its own cutoff, so a caller that names a stat without a threshold still lands on the
    calibrated pair rather than the mean's.

    ``refractory_s`` is the minimum separation between two proposals for THIS goal. It is the
    cadence the gate already enforces (``WatchCondition.cadence_s``), so it is shared with every
    other admission path for the goal and cannot be double-spent.
    """

    goal_id: str
    condition: str
    threshold: float | None = None
    salience_stat: str | None = None
    refractory_s: float = HOSTED_BYOK_DEFAULT_REFRACTORY_S
    region_hint: Box | None = None
    ego_compensate: bool | None = None

    def __post_init__(self) -> None:
        for name in ("goal_id", "condition"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise HostedByokContractError(
                    f"detection goal {name} must be a non-empty, whitespace-trimmed string."
                )
        if self.threshold is not None and (
            type(self.threshold) not in (int, float)
            or not math.isfinite(float(self.threshold))
            or not 0.0 <= float(self.threshold) <= 1.0
        ):
            raise HostedByokContractError("detection goal threshold must be a number in 0..1.")
        if self.salience_stat is not None and self.salience_stat not in _SALIENCE_STATS:
            raise HostedByokContractError(
                f"detection goal salience_stat must be one of {sorted(_SALIENCE_STATS)!r}."
            )
        if (
            type(self.refractory_s) not in (int, float)
            or not math.isfinite(float(self.refractory_s))
            or float(self.refractory_s) <= 0
        ):
            raise HostedByokContractError(
                "detection goal refractory_s must be a finite positive number; an unbounded "
                "detection run cannot state its own worst case."
            )
        if self.region_hint is not None and type(self.region_hint) is not Box:
            raise HostedByokContractError("detection goal region_hint must be an exact Box.")
        if self.ego_compensate is not None and type(self.ego_compensate) is not bool:
            raise HostedByokContractError("detection goal ego_compensate must be an exact bool.")


@dataclass(frozen=True, slots=True)
class HostedByokDetectionProposal:
    """One gated moment: a goal, the exact frame that admitted it, and why.

    The temporal identity (``frame_index`` / ``offset_s`` / ``timestamp_unix_s``) is copied
    verbatim off the admitting :class:`HostedByokGateFrame`, so a hosted incident binds to a
    moment in the caller's own clip coordinates with nothing inferred in between.
    """

    goal_id: str
    frame_index: int
    offset_s: float
    timestamp_unix_s: float | None
    salience: float
    threshold: float
    cause: str
    detector_id: str
    salience_stat: str
    contract_version: str = HOSTED_BYOK_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-able scalars for the hosted wire."""

        return {
            "goal_id": self.goal_id,
            "frame_index": self.frame_index,
            "offset_s": self.offset_s,
            "timestamp_unix_s": self.timestamp_unix_s,
            "salience": self.salience,
            "threshold": self.threshold,
            "cause": self.cause,
            "detector_id": self.detector_id,
            "salience_stat": self.salience_stat,
            "contract_version": self.contract_version,
        }


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


@dataclass(frozen=True, slots=True)
class _JsonAuthority:
    dumps: Callable[..., str]
    loads: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class _Base64Authority:
    b64encode: Callable[..., bytes]
    b64decode: Callable[..., bytes]


@dataclass(frozen=True, slots=True)
class _MathAuthority:
    isfinite: Callable[[float], bool]


def _make_private_wire_codec() -> tuple[
    Callable[[Any], bytes],
    Callable[[bytes], Any],
    Callable[[Any], Any],
]:
    """Recursively freeze the canonical codec instead of retaining live module globals."""

    json_authority = _JsonAuthority(json.dumps, json.loads)
    base64_authority = _Base64Authority(base64.b64encode, base64.b64decode)
    math_authority = _MathAuthority(math.isfinite)

    def clone(
        implementation: Callable[..., Any],
        globals_snapshot: dict[str, Any],
    ) -> Callable[..., Any]:
        frozen = FunctionType(
            implementation.__code__,
            globals_snapshot,
            name=implementation.__name__,
            argdefs=implementation.__defaults__,
            closure=implementation.__closure__,
        )
        frozen.__kwdefaults__ = implementation.__kwdefaults__
        return frozen

    value_globals = dict(_wire_value.__globals__)
    value_globals.update(
        {
            "json": json_authority,
            "base64": base64_authority,
            "math": math_authority,
        }
    )
    frozen_value = clone(_wire_value, value_globals)
    value_globals["_wire_value"] = frozen_value

    reject_globals = dict(_reject_duplicate_json_pairs.__globals__)
    frozen_reject = clone(_reject_duplicate_json_pairs, reject_globals)

    dumps_globals = dict(_wire_dumps.__globals__)
    dumps_globals.update(
        {
            "json": json_authority,
            "_wire_value": frozen_value,
        }
    )
    frozen_dumps = clone(_wire_dumps, dumps_globals)

    unvalue_globals = dict(_wire_unvalue.__globals__)
    unvalue_globals.update(
        {
            "base64": base64_authority,
            "_wire_dumps": frozen_dumps,
        }
    )
    frozen_unvalue = clone(_wire_unvalue, unvalue_globals)
    unvalue_globals["_wire_unvalue"] = frozen_unvalue

    loads_globals = dict(_wire_loads.__globals__)
    loads_globals.update(
        {
            "json": json_authority,
            "_reject_duplicate_json_pairs": frozen_reject,
            "_wire_unvalue": frozen_unvalue,
            "_wire_dumps": frozen_dumps,
        }
    )
    frozen_loads = clone(_wire_loads, loads_globals)
    return frozen_dumps, frozen_loads, frozen_value


def _make_private_input_validator(
    wire_value: Callable[[Any], Any],
) -> Callable[..., None]:
    """Freeze every recursive/helper edge used by hosted call-input validation."""

    def clone(
        implementation: Callable[..., Any],
        globals_snapshot: dict[str, Any],
    ) -> Callable[..., Any]:
        frozen = FunctionType(
            implementation.__code__,
            globals_snapshot,
            name=implementation.__name__,
            argdefs=implementation.__defaults__,
            closure=implementation.__closure__,
        )
        frozen.__kwdefaults__ = implementation.__kwdefaults__
        return frozen

    contains_globals = dict(_input_contains_credential.__globals__)
    frozen_contains = clone(_input_contains_credential, contains_globals)
    contains_globals["_input_contains_credential"] = frozen_contains

    validate_globals = dict(_validate_hosted_call_inputs.__globals__)
    validate_globals.update(
        {
            "math": _MathAuthority(math.isfinite),
            "_wire_value": wire_value,
            "_input_contains_credential": frozen_contains,
        }
    )
    return clone(_validate_hosted_call_inputs, validate_globals)


class HostedByokSubprocessSupervisor:
    """Exact, immutable authenticated no-descendant supervisor for production Hosted BYOK calls."""

    __slots__ = (
        "_poll_interval_s",
        "_boundary_kind",
        "_dispatch",
        "_execute_impl",
        "_locked",
    )

    def __init__(self, *, poll_interval_s: float = 0.002) -> None:
        if (
            type(poll_interval_s) not in (int, float)
            or not math.isfinite(float(poll_interval_s))
            or float(poll_interval_s) <= 0
        ):
            raise HostedByokContractError("poll_interval_s must be a finite positive number.")
        dispatch = _make_private_supervisor_dispatch(float(poll_interval_s))
        object.__setattr__(self, "_poll_interval_s", float(poll_interval_s))
        object.__setattr__(self, "_boundary_kind", dispatch.boundary_kind)
        object.__setattr__(self, "_dispatch", dispatch)
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
        dispatch = self._dispatch
        deadline_monotonic = dispatch.positive_duration(
            "deadline_monotonic", deadline_monotonic
        )
        dispatch.validate_inputs(
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
        await dispatch.sleep(0)
        if request.cancellation is not None and request.cancellation.is_set():
            raise HostedByokCancelled()
        if dispatch.monotonic() >= deadline_monotonic:
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
        worker_payload = dispatch.wire_dumps(worker_values)
        await dispatch.sleep(0)
        if request.cancellation is not None and request.cancellation.is_set():
            raise HostedByokCancelled()
        if dispatch.monotonic() >= deadline_monotonic:
            raise HostedByokDeadlineExceeded(
                "hosted BYOK invocation exceeded its effective deadline."
            )
        outcome = await dispatch.exchange(
            worker_payload,
            request.cancellation,
            deadline_monotonic,
        )
        if request.cancellation is not None and request.cancellation.is_set():
            raise HostedByokCancelled()
        if dispatch.monotonic() >= deadline_monotonic:
            raise HostedByokDeadlineExceeded(
                "hosted BYOK invocation exceeded its effective deadline."
            )
        if type(outcome) is not tuple or len(outcome) != 2:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        kind, payload = outcome
        if type(kind) is not str:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        if kind == "ok" and type(payload) is Verdict:
            if not dispatch.verdict_is_credential_free(payload, request.credential):
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

    def build_info(self) -> dict[str, Any]:
        """Report the hosted contract: version, declared capabilities, and detection bounds."""

        return _hosted_byok_build_info()

    def stream_gated_detections(
        self,
        request: HostedByokRequest,
        goals: Sequence[HostedByokDetectionGoal],
        frames: Sequence[HostedByokGateFrame],
        *,
        settings: Settings | None = None,
        structural_adapter: Callable[[Any], float] | None = None,
        semantic_adapter: Callable[[Any], float] | None = None,
    ) -> AsyncIterator[HostedByokDetectionProposal]:
        """Stream gated proposals for a long clip without materializing them all first.

        The whole route is preflighted before the returned iterator yields anything, so an
        unentitled provider, an unsupported pinned contract version, or a malformed frame sequence
        fails at preflight rather than partway through a clip. Exhausting the iterator with zero
        proposals raises :class:`HostedByokDetectionStarved` — see :meth:`detect_gated`.
        """

        if type(self) is not HostedByokAdapterFactory:
            raise HostedByokContractError("factory must be the exact SDK HostedByokAdapterFactory.")
        if type(request) is not HostedByokRequest:
            raise HostedByokContractError(
                "request must be an immutable HostedByokRequest; implicit configuration is "
                "unsupported."
            )
        for name, adapter in (
            ("structural_adapter", structural_adapter),
            ("semantic_adapter", semantic_adapter),
        ):
            if adapter is not None and not callable(adapter):
                raise HostedByokContractError(f"hosted BYOK detection {name} must be callable.")
        entered_monotonic = float(self._monotonic_fn())
        deadline_monotonic = self._deadline_at_entry(request, entered_monotonic=entered_monotonic)
        if request.cancellation is not None and request.cancellation.is_set():
            raise HostedByokCancelled()
        plan = _preflight_detection(request, goals, frames, settings=settings)
        return _gated_detection_stream(
            request,
            plan,
            monotonic_fn=self._monotonic_fn,
            deadline_monotonic=deadline_monotonic,
            structural_adapter=structural_adapter,
            semantic_adapter=semantic_adapter,
        )

    async def detect_gated(
        self,
        request: HostedByokRequest,
        goals: Sequence[HostedByokDetectionGoal],
        frames: Sequence[HostedByokGateFrame],
        *,
        settings: Settings | None = None,
        structural_adapter: Callable[[Any], float] | None = None,
        semantic_adapter: Callable[[Any], float] | None = None,
    ) -> tuple[HostedByokDetectionProposal, ...]:
        """Return ZERO OR MORE gated proposals — the moments worth judging — in clip order.

        This route spends **no provider call and reads no credential**: it is the tier-0 salience
        gate run over gate-resolution frames, and its whole output is temporal identity plus the
        reason a moment was admitted. The judgment of an admitted moment is still :meth:`judge`.

        A run that admits nothing raises :class:`HostedByokDetectionStarved` rather than returning
        an empty tuple, because an empty list cannot be told apart from "nothing happened" by a
        caller who never sees the salience distribution.
        """

        proposals: list[HostedByokDetectionProposal] = []
        stream = self.stream_gated_detections(
            request,
            goals,
            frames,
            settings=settings,
            structural_adapter=structural_adapter,
            semantic_adapter=semantic_adapter,
        )
        try:
            async for proposal in stream:
                proposals.append(proposal)
        finally:
            await stream.aclose()
        return tuple(proposals)

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
        introduced_in = _CAPABILITY_MIN_CONTRACT_VERSION.get(request.capability)
        if introduced_in is not None and _version_below(request.contract_version, introduced_in):
            raise HostedByokUnsupportedError(
                f"capability {request.capability!r} was introduced in hosted BYOK contract "
                f"{introduced_in}; this request pins {request.contract_version}."
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


# ---------------------------------------------------------------------------
# vision.detect.gated — the engine
# ---------------------------------------------------------------------------

_DETECTION_CAPABILITY = "vision.detect.gated"
_SALIENCE_STATS = frozenset({"mean", "block_max", "p95"})
#: Hard ceiling on the frames one detection call accepts. 1,200 gate frames is 600 s at 2 fps, and
#: the byte bound below is the real limit; this one exists so a malformed request is refused before
#: a large sequence is walked at all.
_MAX_DETECTION_FRAMES = 200_000
_MAX_DETECTION_GOALS = 64


def _version_below(version: str, minimum: str) -> bool:
    """Compare two exact ``major.minor.patch`` contract versions. Both are validated literals."""

    def parts(value: str) -> tuple[int, ...]:
        return tuple(int(piece) for piece in value.split("."))

    return parts(version) < parts(minimum)


_CREDENTIAL_FIREWALL_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "hosted_byok_credential_firewall_depth", default=0
)
_CREDENTIAL_FIREWALL_INSTALLS = 0
_CREDENTIAL_FIREWALL_REAL_ENVIRON: Any = None


class _CredentialFirewallEnviron(MutableMapping):
    """An ``os.environ`` stand-in that refuses provider keys inside a detection scope.

    Gated detection contacts no provider, so no provider credential may be read on its path — and
    "we simply do not call ``os.environ``" is a claim that decays the moment someone adds a helper.
    This makes it enforced rather than asserted: while a detection scope is active *on the reading
    task's own context*, a read of any key in :data:`_PROVIDER_ENV_KEYS` raises. The scope is a
    :class:`contextvars.ContextVar`, so an unrelated coroutine interleaving on the same loop reads
    the real environment untouched, and Cloud's own tasks are unaffected.

    Caller-supplied salience adapters run inside the scope too: a hosted tenant's own tier-0
    channel cannot be used as a credential-exfiltration path either.
    """

    __slots__ = ("_real",)

    def __init__(self, real: Any) -> None:
        self._real = real

    @staticmethod
    def _check(key: Any) -> None:
        if _CREDENTIAL_FIREWALL_DEPTH.get() and key in _PROVIDER_ENV_KEYS:
            raise HostedByokContractError(
                "hosted BYOK gated detection contacts no provider and must not read provider "
                f"configuration or credentials from the environment; blocked read of {key!r}."
            )

    def __getitem__(self, key: Any) -> Any:
        self._check(key)
        return self._real[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._real[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._real[key]

    def __iter__(self) -> Any:
        return iter(self._real)

    def __len__(self) -> int:
        return len(self._real)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return repr(self._real)

    def copy(self) -> dict[str, str]:
        return dict(self._real)

    def setdefault(self, key: Any, value: Any = None) -> Any:  # type: ignore[override]
        self._check(key)
        return self._real.setdefault(key, value)

    def __or__(self, other: Any) -> Any:  # pragma: no cover - parity with os._Environ
        return self._real | other

    def __ror__(self, other: Any) -> Any:  # pragma: no cover - parity with os._Environ
        return other | self._real


class _ProviderCredentialFirewall:
    """Enter/exit the credential firewall for exactly one detection scope."""

    __slots__ = ("_token", "_installed")

    def __init__(self) -> None:
        self._token: contextvars.Token[int] | None = None
        self._installed = False

    def __enter__(self) -> "_ProviderCredentialFirewall":
        global _CREDENTIAL_FIREWALL_INSTALLS, _CREDENTIAL_FIREWALL_REAL_ENVIRON
        if not isinstance(os.environ, _CredentialFirewallEnviron):
            _CREDENTIAL_FIREWALL_REAL_ENVIRON = os.environ
            os.environ = _CredentialFirewallEnviron(_CREDENTIAL_FIREWALL_REAL_ENVIRON)  # type: ignore[assignment]
        _CREDENTIAL_FIREWALL_INSTALLS += 1
        self._installed = True
        self._token = _CREDENTIAL_FIREWALL_DEPTH.set(_CREDENTIAL_FIREWALL_DEPTH.get() + 1)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        global _CREDENTIAL_FIREWALL_INSTALLS, _CREDENTIAL_FIREWALL_REAL_ENVIRON
        if self._token is not None:
            _CREDENTIAL_FIREWALL_DEPTH.reset(self._token)
            self._token = None
        if not self._installed:
            return
        self._installed = False
        _CREDENTIAL_FIREWALL_INSTALLS -= 1
        if _CREDENTIAL_FIREWALL_INSTALLS <= 0:
            _CREDENTIAL_FIREWALL_INSTALLS = 0
            if _CREDENTIAL_FIREWALL_REAL_ENVIRON is not None and isinstance(
                os.environ, _CredentialFirewallEnviron
            ):
                os.environ = _CREDENTIAL_FIREWALL_REAL_ENVIRON  # type: ignore[assignment]
            _CREDENTIAL_FIREWALL_REAL_ENVIRON = None


@dataclass(frozen=True, slots=True)
class _DetectionPlan:
    """The validated, fully-resolved detection route — everything the scan needs, nothing else."""

    conditions: tuple[Any, ...]
    thresholds: Mapping[str, float]
    #: The resolved salience statistic per goal id — a call may mix statistics across goals.
    stats: Mapping[str, str]
    #: The single statistic when every goal resolved to the same one, else ``"mixed"``. Used only
    #: for the run-level starvation diagnostic, never for a per-proposal claim.
    salience_stat: str
    settings: Settings
    frames: tuple[HostedByokGateFrame, ...]
    goal_ids: tuple[str, ...]


def _resolve_detection_threshold(stat: str, settings: Settings) -> float:
    """The calibrated base cutoff for one salience statistic.

    Mirrors ``percept.harness.tier0.compile_watchspec``'s ``stat_thresholds`` rule EXACTLY: the
    localized statistics sit on a higher value scale than the whole-frame mean and carry their own
    calibrated cutoffs, so naming a stat without a threshold must not silently apply the mean's.
    """

    return {
        "mean": settings.edge.motion_gate,
        "block_max": settings.edge.motion_gate_block_max,
        "p95": settings.edge.motion_gate_p95,
    }.get(stat, settings.edge.motion_gate)


def _preflight_detection(
    request: HostedByokRequest,
    goals: Sequence[HostedByokDetectionGoal],
    frames: Sequence[HostedByokGateFrame],
    *,
    settings: Settings | None,
) -> _DetectionPlan:
    """Validate the whole route before a single pixel is read.

    Everything refusable is refused HERE: the provider's declared capability, the pinned contract
    version, the goal shapes, and the frame geometry/ordering/size. A caller that is not entitled
    to gated detection learns so at preflight, not after walking a 600 s clip.
    """

    from .harness.tier0 import Frame, WatchCondition  # noqa: PLC0415

    HostedByokAdapterFactory._validate_route(request)
    capabilities = HOSTED_BYOK_CAPABILITIES.get(request.provider) or frozenset()
    if _DETECTION_CAPABILITY not in capabilities:
        raise HostedByokUnsupportedError(
            f"provider {request.provider!r} does not declare capability "
            f"{_DETECTION_CAPABILITY!r}; supported capabilities are {sorted(capabilities)!r}."
        )
    introduced_in = _CAPABILITY_MIN_CONTRACT_VERSION[_DETECTION_CAPABILITY]
    if _version_below(request.contract_version, introduced_in):
        raise HostedByokUnsupportedError(
            f"capability {_DETECTION_CAPABILITY!r} was introduced in hosted BYOK contract "
            f"{introduced_in}; this request pins {request.contract_version}."
        )
    if settings is not None and not isinstance(settings, Settings):
        raise HostedByokContractError("hosted BYOK detection settings must be a Settings.")
    # Resolved BEFORE the credential firewall arms: deployment configuration is read from the
    # environment by design, a provider credential never is.
    resolved = load_settings() if settings is None else settings

    if type(goals) is not tuple and type(goals) is not list:
        raise HostedByokContractError("hosted BYOK detection goals must be an exact list/tuple.")
    if not goals:
        raise HostedByokContractError(
            "hosted BYOK detection requires at least one goal; an empty goal set can only ever "
            "admit nothing."
        )
    if len(goals) > _MAX_DETECTION_GOALS:
        raise HostedByokContractError(
            f"hosted BYOK detection accepts at most {_MAX_DETECTION_GOALS} goals per call."
        )
    seen: set[str] = set()
    conditions: list[Any] = []
    thresholds: dict[str, float] = {}
    stats: dict[str, str] = {}
    for goal in goals:
        if type(goal) is not HostedByokDetectionGoal:
            raise HostedByokContractError(
                "hosted BYOK detection goals must be exact HostedByokDetectionGoal values."
            )
        if goal.goal_id in seen:
            raise HostedByokContractError(
                f"hosted BYOK detection goal id {goal.goal_id!r} appears more than once."
            )
        seen.add(goal.goal_id)
        stat = goal.salience_stat or resolved.edge.salience_stat
        if stat not in _SALIENCE_STATS:
            raise HostedByokContractError(
                f"deployment salience_stat {stat!r} is outside {sorted(_SALIENCE_STATS)!r}."
            )
        threshold = (
            _resolve_detection_threshold(stat, resolved)
            if goal.threshold is None
            else float(goal.threshold)
        )
        threshold = 0.0 if threshold < 0.0 else 1.0 if threshold > 1.0 else threshold
        thresholds[goal.goal_id] = threshold
        stats[goal.goal_id] = stat
        conditions.append(
            WatchCondition(
                goal_id=goal.goal_id,
                anchor_text=goal.condition,
                threshold=threshold,
                region_hint=goal.region_hint,
                # No cheap edge detectors: this route proposes gate admissions, never boundaries.
                detectors=[],
                schedule=None,
                # The refractory IS the gate's own per-goal cadence, shared with every other
                # admission path for the goal, so a proposal cannot double-spend inside it.
                cadence_s=float(goal.refractory_s),
                gate_on="motion",
                salience_stat=stat,
                ego_compensate=(
                    resolved.edge.ego_compensate
                    if goal.ego_compensate is None
                    else goal.ego_compensate
                ),
            )
        )

    if type(frames) is not tuple and type(frames) is not list:
        raise HostedByokContractError("hosted BYOK detection frames must be an exact list/tuple.")
    if not frames:
        raise HostedByokContractError("hosted BYOK detection requires at least one gate frame.")
    if len(frames) > _MAX_DETECTION_FRAMES:
        raise HostedByokContractError(
            f"hosted BYOK detection accepts at most {_MAX_DETECTION_FRAMES} gate frames per call."
        )
    first = frames[0]
    if type(first) is not HostedByokGateFrame:
        raise HostedByokContractError(
            "hosted BYOK detection frames must be exact HostedByokGateFrame values."
        )
    total_bytes = len(frames) * first.width * first.height
    if total_bytes > _MAX_PROCESS_IPC_BYTES:
        raise HostedByokContractError(
            f"hosted BYOK detection payload of {total_bytes} luma bytes exceeds the "
            f"{_MAX_PROCESS_IPC_BYTES}-byte boundary ceiling; split the clip."
        )
    previous: HostedByokGateFrame | None = None
    for frame in frames:
        if type(frame) is not HostedByokGateFrame:
            raise HostedByokContractError(
                "hosted BYOK detection frames must be exact HostedByokGateFrame values."
            )
        if frame.width != first.width or frame.height != first.height:
            raise HostedByokContractError(
                "hosted BYOK detection frames must all share one exact gate resolution; "
                f"got {frame.width}x{frame.height} after {first.width}x{first.height}."
            )
        if previous is not None:
            if frame.frame_index <= previous.frame_index:
                raise HostedByokContractError(
                    "hosted BYOK detection frames must carry strictly increasing frame_index."
                )
            if float(frame.offset_s) < float(previous.offset_s):
                raise HostedByokContractError(
                    "hosted BYOK detection frames must be ordered by non-decreasing offset_s; "
                    "a gate cannot run time backwards."
                )
        previous = frame
        # Reject anything the gate itself would reject, at preflight rather than mid-scan.
        Frame(w=frame.width, h=frame.height, pixels=frame.pixels)

    distinct_stats = set(stats.values())
    return _DetectionPlan(
        conditions=tuple(conditions),
        thresholds=MappingProxyType(dict(thresholds)),
        stats=MappingProxyType(dict(stats)),
        salience_stat=next(iter(distinct_stats)) if len(distinct_stats) == 1 else "mixed",
        settings=resolved,
        frames=tuple(frames),
        goal_ids=tuple(goal.goal_id for goal in goals),
    )


async def _gated_detection_stream(
    request: HostedByokRequest,
    plan: _DetectionPlan,
    *,
    monotonic_fn: Callable[[], float],
    deadline_monotonic: float | None,
    structural_adapter: Callable[[Any], float] | None,
    semantic_adapter: Callable[[Any], float] | None,
) -> AsyncIterator[HostedByokDetectionProposal]:
    """Yield one proposal per gate admission, in clip order.

    Reuses :class:`percept.harness.tier0.MotionGate` — the SAME gate the runtime escalates on. No
    second gate exists in this module and none may be added: two gates drift, and a hosted tenant
    would then be attributing incidents with different admission semantics than the SDK's own.

    The scan is synchronous pure arithmetic over a tiny buffer, so it is NOT offloaded to a thread
    or executor: it yields to the loop between frames instead. That keeps cancellation genuine
    (a cancelled task stops between two frames, owning nothing) and spawns no child process and no
    thread that a caller's own pool would have to join.
    """

    from .harness.tier0 import Frame, MotionGate, WatchSpec  # noqa: PLC0415

    spec = WatchSpec(conditions=list(plan.conditions))
    gate = MotionGate(
        spec,
        settings=plan.settings,
        structural_adapter=structural_adapter,
        semantic_adapter=semantic_adapter,
    )
    admitted = 0
    max_salience = 0.0
    salience_total = 0.0
    salience_samples = 0
    cancellation = request.cancellation
    with _ProviderCredentialFirewall():
        for gate_frame in plan.frames:
            # One real cancellation point per frame. asyncio.sleep(0) reschedules this coroutine,
            # so a cancel between frames raises here and unwinds with no partial state retained.
            await asyncio.sleep(0)
            if cancellation is not None and cancellation.is_set():
                raise HostedByokCancelled(
                    "hosted BYOK gated detection was cancelled by its invocation signal."
                )
            if deadline_monotonic is not None and float(monotonic_fn()) >= deadline_monotonic:
                raise HostedByokDeadlineExceeded(
                    "hosted BYOK gated detection deadline elapsed mid-clip."
                )
            decision = gate.evaluate(
                Frame(w=gate_frame.width, h=gate_frame.height, pixels=gate_frame.pixels),
                ts=float(gate_frame.offset_s),
                include_outcomes=True,
            )
            by_goal = {outcome.goal_id: outcome for outcome in decision.outcomes}
            for outcome in decision.outcomes:
                salience_total += float(outcome.score)
                salience_samples += 1
                if float(outcome.score) > max_salience:
                    max_salience = float(outcome.score)
            for goal_id, cause in zip(decision.goal_ids, decision.admission_causes):
                outcome = by_goal[goal_id]
                admitted += 1
                yield HostedByokDetectionProposal(
                    goal_id=goal_id,
                    frame_index=gate_frame.frame_index,
                    offset_s=float(gate_frame.offset_s),
                    timestamp_unix_s=(
                        None
                        if gate_frame.timestamp_unix_s is None
                        else float(gate_frame.timestamp_unix_s)
                    ),
                    salience=float(outcome.score),
                    threshold=float(outcome.threshold),
                    cause=str(cause),
                    detector_id=outcome.detector_id,
                    salience_stat=plan.stats[goal_id],
                    contract_version=request.contract_version,
                )
    if admitted == 0:
        mean_salience = salience_total / salience_samples if salience_samples else 0.0
        lowest = min(plan.thresholds.values())
        # "Real signal that never cleared the bar" is a configuration failure; "no signal at all"
        # is a still clip. Both are reported, because only the caller knows which it expected.
        misconfigured = max_salience > 0.0 and max_salience >= lowest * 0.25
        raise HostedByokDetectionStarved(
            "hosted BYOK gated detection admitted nothing across the whole run: "
            f"{len(plan.frames)} frame(s), {plan.salience_stat} salience "
            f"max={max_salience:.6g} mean={mean_salience:.6g}, lowest threshold={lowest:.6g}. "
            + (
                "The gate observed real salience that never cleared its cutoff — retune the "
                "threshold/statistic or supply an admission signal."
                if misconfigured
                else "The gate observed essentially no salience — the clip was still, or the "
                "frames are not the tier-0 gate downsample."
            ),
            goal_ids=plan.goal_ids,
            frames=len(plan.frames),
            salience_stat=plan.salience_stat,
            thresholds=plan.thresholds,
            max_salience=max_salience,
            mean_salience=mean_salience,
            misconfigured=misconfigured,
        )


def _hosted_byok_build_info() -> dict[str, Any]:
    """The hosted contract descriptor Cloud preflights against."""

    try:
        from ._build import build_info as _package_build_info  # noqa: PLC0415

        package = _package_build_info().to_dict()
    except Exception:  # pragma: no cover - defensive; build_info never raises
        package = {}
    return {
        "contract_version": HOSTED_BYOK_CONTRACT_VERSION,
        "supported_contract_versions": list(HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS),
        "process_isolation_version": HOSTED_BYOK_PROCESS_ISOLATION_VERSION,
        "capabilities": {
            provider: sorted(caps) for provider, caps in HOSTED_BYOK_CAPABILITIES.items()
        },
        "capability_min_contract_version": dict(_CAPABILITY_MIN_CONTRACT_VERSION),
        "detection": {
            "capability": _DETECTION_CAPABILITY,
            "default_refractory_s": HOSTED_BYOK_DEFAULT_REFRACTORY_S,
            "max_gate_frame_pixels": HOSTED_BYOK_MAX_GATE_FRAME_PIXELS,
            "max_gate_frames": _MAX_DETECTION_FRAMES,
            "max_goals": _MAX_DETECTION_GOALS,
            "max_request_bytes": _MAX_PROCESS_IPC_BYTES,
            "spends_provider_calls": False,
        },
        "package": package,
    }


@dataclass(frozen=True, slots=True, repr=False)
class HostedByokVision:
    """Request-bound Vision adapter for direct use with :class:`percept.Percept`.

    The adapter holds only the immutable request and factory; every ``judge`` delegates to a fresh
    invocation through the factory.

    ``detect``/``detect_events`` are the ``Vision`` protocol's OBJECT-detection surface — "what
    objects are in this frame", answered by a provider. This boundary still answers them with no
    proposals, honestly: it exposes no per-frame object detector.

    :meth:`detect_gated` is a different question and a different route (contract 1.1,
    ``vision.detect.gated``): "WHEN in this clip is a moment worth judging". It runs the tier-0
    salience gate over gate-resolution frames, contacts no provider, and returns zero or more
    moments carrying temporal identity — the thing a hosted incident binds to.
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

    def build_info(self) -> dict[str, Any]:
        """The hosted contract descriptor plus this binding's exact route."""

        info = self._factory.build_info()
        info["request"] = {
            "provider": self._request.provider,
            "model": self._request.model,
            "capability": self._request.capability,
            "contract_version": self._request.contract_version,
            "credential_lease_id": self._request.credential_lease_id,
            "principal_id": self._request.principal_id,
        }
        return info

    async def detect_gated(
        self,
        goals: Sequence[HostedByokDetectionGoal],
        frames: Sequence[HostedByokGateFrame],
        *,
        settings: Settings | None = None,
        structural_adapter: Callable[[Any], float] | None = None,
        semantic_adapter: Callable[[Any], float] | None = None,
    ) -> tuple[HostedByokDetectionProposal, ...]:
        """Return the gated moments in ``frames`` worth judging — see
        :meth:`HostedByokAdapterFactory.detect_gated`."""

        return await self._factory.detect_gated(
            self._request,
            goals,
            frames,
            settings=settings,
            structural_adapter=structural_adapter,
            semantic_adapter=semantic_adapter,
        )

    def stream_gated_detections(
        self,
        goals: Sequence[HostedByokDetectionGoal],
        frames: Sequence[HostedByokGateFrame],
        *,
        settings: Settings | None = None,
        structural_adapter: Callable[[Any], float] | None = None,
        semantic_adapter: Callable[[Any], float] | None = None,
    ) -> AsyncIterator[HostedByokDetectionProposal]:
        """Stream the gated moments in ``frames`` — see
        :meth:`HostedByokAdapterFactory.stream_gated_detections`."""

        return self._factory.stream_gated_detections(
            self._request,
            goals,
            frames,
            settings=settings,
            structural_adapter=structural_adapter,
            semantic_adapter=semantic_adapter,
        )

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


@dataclass(frozen=True, slots=True)
class _PrivateSupervisorDispatch:
    """Per-supervisor launch and I/O authority captured before caller code can poison globals."""

    boundary_kind: str
    exchange: Callable[[bytes, asyncio.Event | None, float], Any]
    wire_dumps: Callable[[Any], bytes]
    positive_duration: Callable[[str, float], float]
    validate_inputs: Callable[..., None]
    verdict_is_credential_free: Callable[[Verdict, str], bool]
    sleep: Callable[[float], Any]
    monotonic: Callable[[], float]


def _make_worker_exchange(
    *,
    launch: Callable[[], tuple[Any, socket.socket, bytes]],
    terminate: Callable[[Any], None],
    verify_ready: Callable[[Any, bytes], bool],
    wire_loads: Callable[[bytes], Any],
    sleep: Callable[[float], Any],
    monotonic: Callable[[], float],
    poll_interval_s: float,
    max_payload_bytes: int,
    invocation_error: type[HostedByokInvocationError],
    deadline_error: type[HostedByokDeadlineExceeded],
    cancellation_error: type[HostedByokCancelled],
    failure_code: str,
) -> Callable[[bytes, asyncio.Event | None, float], Any]:
    """Seal one nonblocking framed exchange; every exit closes and joins the worker."""

    header = struct.Struct("!I")

    def fail() -> None:
        raise invocation_error(failure_code)

    async def checkpoint(
        cancellation: asyncio.Event | None,
        deadline_monotonic: float,
    ) -> None:
        if cancellation is not None and cancellation.is_set():
            raise cancellation_error()
        if monotonic() >= deadline_monotonic:
            raise deadline_error(
                "hosted BYOK invocation exceeded its effective deadline."
            )

    async def receive_frame(
        peer: socket.socket,
        process: Any,
        cancellation: asyncio.Event | None,
        deadline_monotonic: float,
    ) -> bytes:
        header_bytes = bytearray()
        body = bytearray()
        expected_size: int | None = None
        while True:
            await checkpoint(cancellation, deadline_monotonic)
            target = header_bytes if expected_size is None else body
            remaining = (
                header.size - len(header_bytes)
                if expected_size is None
                else expected_size - len(body)
            )
            if remaining == 0:
                if expected_size is None:
                    expected_size = header.unpack(bytes(header_bytes))[0]
                    if expected_size > max_payload_bytes:
                        fail()
                    if expected_size == 0:
                        return b""
                    continue
                return bytes(body)
            try:
                chunk = peer.recv(remaining)
            except BlockingIOError:
                chunk = None
            except (InterruptedError, OSError):
                fail()
            if chunk is None:
                if process.poll() is not None:
                    fail()
                await sleep(poll_interval_s)
                continue
            if not chunk:
                fail()
            target.extend(chunk)

    async def send_frame(
        peer: socket.socket,
        process: Any,
        payload: bytes,
        cancellation: asyncio.Event | None,
        deadline_monotonic: float,
    ) -> None:
        if type(payload) is not bytes or len(payload) > max_payload_bytes:
            fail()
        framed = header.pack(len(payload)) + payload
        view = memoryview(framed)
        sent = 0
        while sent < len(view):
            await checkpoint(cancellation, deadline_monotonic)
            try:
                count = peer.send(view[sent:])
            except BlockingIOError:
                count = None
            except (InterruptedError, OSError):
                fail()
            if count is None:
                if process.poll() is not None:
                    fail()
                await sleep(poll_interval_s)
                continue
            if count <= 0:
                fail()
            sent += count

    async def exchange(
        worker_payload: bytes,
        cancellation: asyncio.Event | None,
        deadline_monotonic: float,
    ) -> Any:
        process, peer, challenge = launch()
        try:
            ready = wire_loads(
                await receive_frame(
                    peer,
                    process,
                    cancellation,
                    deadline_monotonic,
                )
            )
            if not verify_ready(ready, challenge):
                fail()
            await checkpoint(cancellation, deadline_monotonic)
            await sleep(0)
            await checkpoint(cancellation, deadline_monotonic)
            await send_frame(
                peer,
                process,
                worker_payload,
                cancellation,
                deadline_monotonic,
            )
            return wire_loads(
                await receive_frame(
                    peer,
                    process,
                    cancellation,
                    deadline_monotonic,
                )
            )
        finally:
            close_failed = False
            try:
                peer.close()
            except BaseException:
                close_failed = True
            finally:
                terminate(process)
            if close_failed:
                fail()

    return exchange


def _make_private_supervisor_dispatch(poll_interval_s: float) -> _PrivateSupervisorDispatch:
    """Capture an absolute launch authority and all mutable module seams exactly once."""

    boundary_kind, selected_prefix = _select_containment_capability()
    prefix = tuple(selected_prefix)
    environment_items = tuple(sorted(_clean_child_environment().items()))
    import_roots_json = json.dumps(
        _clean_child_import_roots(),
        separators=(",", ":"),
    )
    selected_machine = platform.machine().lower()
    popen_type = subprocess.Popen
    socketpair = socket.socketpair
    urandom = os.urandom
    stat_file = os.stat
    realpath = os.path.realpath
    is_absolute = os.path.isabs
    is_regular = stat.S_ISREG
    mode_world_write = stat.S_IWOTH
    mode_group_write = stat.S_IWGRP
    mode_execute = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    interpreter_path = realpath(sys.executable)
    child_bootstrap = _CHILD_BOOTSTRAP
    devnull = subprocess.DEVNULL
    failure_code = _PROVIDER_FAILURE_CODE
    isolation_version = HOSTED_BYOK_PROCESS_ISOLATION_VERSION
    compare_digest = hmac.compare_digest
    invocation_error = HostedByokInvocationError
    contract_error = HostedByokContractError
    wire_dumps, wire_loads, wire_value = _make_private_wire_codec()
    validate_inputs = _make_private_input_validator(wire_value)

    def file_attestation(path: str, *, trusted_tool: bool) -> tuple[Any, ...]:
        resolved = realpath(path)
        try:
            metadata = stat_file(resolved, follow_symlinks=False)
        except OSError:
            raise contract_error(
                "hosted BYOK trusted process executable is unavailable."
            ) from None
        if (
            not is_absolute(resolved)
            or not is_regular(metadata.st_mode)
            or metadata.st_mode & mode_world_write
            or not metadata.st_mode & mode_execute
            or (
                trusted_tool
                and (
                    metadata.st_uid != 0
                    or metadata.st_mode & (mode_group_write | mode_world_write)
                )
            )
        ):
            raise contract_error(
                "hosted BYOK trusted process executable failed identity validation."
            )
        return (
            "executable-v1",
            resolved,
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_size),
            int(metadata.st_uid),
            int(metadata.st_mode & 0o7777),
        )

    interpreter_attestation = file_attestation(interpreter_path, trusted_tool=False)
    tool_attestations = tuple(
        file_attestation(item, trusted_tool=True)
        for item in prefix
        if item.startswith("/")
    )

    def attest_files_unchanged() -> None:
        if file_attestation(interpreter_path, trusted_tool=False) != interpreter_attestation:
            raise contract_error(
                "hosted BYOK interpreter identity changed after supervisor construction."
            )
        current_tools = tuple(
            file_attestation(item, trusted_tool=True)
            for item in prefix
            if item.startswith("/")
        )
        if current_tools != tool_attestations:
            raise contract_error(
                "hosted BYOK containment tool identity changed after supervisor construction."
            )

    def launch() -> tuple[Any, socket.socket, bytes]:
        attest_files_unchanged()
        challenge = urandom(32)
        if type(challenge) is not bytes or len(challenge) != 32:
            raise invocation_error(failure_code)
        parent_socket, child_socket = socketpair()

        def close_socket_pair() -> None:
            # Each close gets a chance, while neither close can prevent process reclamation.
            try:
                try:
                    parent_socket.close()
                except BaseException:
                    pass
            finally:
                try:
                    child_socket.close()
                except BaseException:
                    pass

        try:
            child_fd = child_socket.fileno()
            command = [
                *prefix,
                interpreter_path,
                "-I",
                "-S",
                "-B",
                "-c",
                child_bootstrap,
                import_roots_json,
                str(child_fd),
                challenge.hex(),
                boundary_kind,
            ]
        except BaseException:
            close_socket_pair()
            raise invocation_error(failure_code) from None

        try:
            process = popen_type(
                command,
                stdin=devnull,
                stdout=devnull,
                stderr=devnull,
                close_fds=True,
                pass_fds=(child_fd,),
                cwd="/",
                env=dict(environment_items),
            )
        except BaseException:
            close_socket_pair()
            raise invocation_error(failure_code) from None

        # Popen returning transfers process ownership here. Every subsequent setup failure
        # closes both sockets and synchronously kills/joins that exact process.
        try:
            parent_socket.setblocking(False)
            child_socket.close()
        except BaseException:
            try:
                close_socket_pair()
            finally:
                terminate(process)
            raise invocation_error(failure_code) from None
        return process, parent_socket, challenge

    def terminate(process: Any) -> None:
        try:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            process.wait()
        except Exception:
            raise invocation_error(failure_code) from None
        if process.poll() is None:
            raise invocation_error(failure_code)

    required_environment = tuple(sorted(environment_items))

    def verify_ready(ready: Any, challenge: bytes) -> bool:
        if (
            type(ready) is not tuple
            or len(ready) != 7
            or ready[0] != "ready"
            or ready[1] != isolation_version
            or type(ready[2]) is not bytes
            or not compare_digest(ready[2], challenge)
            or ready[3] != boundary_kind
            or ready[4] != interpreter_attestation
            or type(ready[5]) is not tuple
            or type(ready[6]) is not tuple
        ):
            return False
        environment_attestation = ready[6]
        if (
            len(environment_attestation) != 3
            or environment_attestation[0] != "environment-v1"
            or environment_attestation[1] != required_environment
            or type(environment_attestation[2]) is not bool
            or (boundary_kind != "darwin_sandbox_no_descendants" and environment_attestation[2])
        ):
            return False
        containment = ready[5]
        if boundary_kind == "darwin_sandbox_no_descendants":
            return containment == (
                "darwin-no-descendants-v1",
                "darwin",
                selected_machine,
                "rlimit-nproc-zero",
                "fork-denied",
                "thread-ok",
            )
        return (
            len(containment) == 10
            and containment[:7]
            == (
                "linux-seccomp-amd64-v1",
                "linux",
                "x86_64",
                "elf64-x86-64",
                1,
                2,
                "fork-eperm",
            )
            and containment[7:9] == ("exec-eperm", "thread-ok")
            and type(containment[9]) is int
            and containment[9] >= 1
        )

    exchange = _make_worker_exchange(
        launch=launch,
        terminate=terminate,
        verify_ready=verify_ready,
        wire_loads=wire_loads,
        sleep=asyncio.sleep,
        monotonic=time.monotonic,
        poll_interval_s=poll_interval_s,
        max_payload_bytes=_MAX_PROCESS_IPC_BYTES,
        invocation_error=HostedByokInvocationError,
        deadline_error=HostedByokDeadlineExceeded,
        cancellation_error=HostedByokCancelled,
        failure_code=failure_code,
    )
    return _PrivateSupervisorDispatch(
        boundary_kind=boundary_kind,
        exchange=exchange,
        wire_dumps=wire_dumps,
        positive_duration=_positive_duration,
        validate_inputs=validate_inputs,
        verdict_is_credential_free=_verdict_is_credential_free,
        sleep=asyncio.sleep,
        monotonic=time.monotonic,
    )


def _scrub_provider_environment() -> None:
    for key in _PROVIDER_ENV_KEYS:
        os.environ.pop(key, None)


def _current_executable_attestation() -> tuple[Any, ...]:
    path = os.path.realpath(sys.executable)
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
    if (
        not os.path.isabs(path)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & stat.S_IWOTH
        or not metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    ):
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    return (
        "executable-v1",
        path,
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_uid),
        int(metadata.st_mode & 0o7777),
    )


def _attest_clean_child_environment() -> tuple[Any, ...]:
    required = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }
    actual = dict(os.environ)
    darwin_text_encoding = actual.pop("__CF_USER_TEXT_ENCODING", None)
    if actual != required:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    has_darwin_text_encoding = darwin_text_encoding is not None
    if has_darwin_text_encoding:
        if sys.platform != "darwin":
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        pieces = darwin_text_encoding.split(":")
        try:
            valid = (
                len(pieces) == 3
                and all(piece.startswith("0x") for piece in pieces)
                and all(int(piece, 16) >= 0 for piece in pieces)
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    return (
        "environment-v1",
        tuple(sorted(required.items())),
        has_darwin_text_encoding,
    )


def _linux_status_values() -> tuple[int, int, int]:
    try:
        with open("/proc/self/status", encoding="ascii") as status_file:
            values = {}
            for line in status_file:
                if ":" in line:
                    key, raw_value = line.split(":", 1)
                    if key in {"NoNewPrivs", "Seccomp", "Seccomp_filters"}:
                        values[key] = int(raw_value.strip())
    except (OSError, UnicodeError, ValueError):
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
    if set(values) != {"NoNewPrivs", "Seccomp", "Seccomp_filters"}:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    return values["NoNewPrivs"], values["Seccomp"], values["Seccomp_filters"]


def _attest_native_linux_amd64() -> tuple[int, int, int]:
    try:
        uname = os.uname()
        with open("/proc/self/exe", "rb") as executable:
            elf_header = executable.read(20)
        with open("/proc/cpuinfo", encoding="ascii", errors="strict") as cpuinfo_file:
            cpuinfo = cpuinfo_file.read(1024 * 1024).lower()
    except (AttributeError, OSError, UnicodeError):
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
    emulation_markers = ("virtualapple", "rosetta", "qemu", "tcg")
    if (
        sys.platform != "linux"
        or uname.sysname != "Linux"
        or uname.machine.lower() != "x86_64"
        or sys.byteorder != "little"
        or struct.calcsize("P") != 8
        or len(elf_header) != 20
        or elf_header[:6] != b"\x7fELF\x02\x01"
        or int.from_bytes(elf_header[18:20], "little") != 62
        or "vendor_id" not in cpuinfo
        or "flags" not in cpuinfo
        or " lm " not in f" {cpuinfo.replace(chr(10), ' ')} "
        or any(marker in cpuinfo for marker in emulation_markers)
    ):
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    return _linux_status_values()


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


def _establish_no_descendant_boundary(boundary_kind: str) -> tuple[Any, ...]:
    """Independently select, install, and attest the exact child-side platform boundary."""

    pre_linux_status: tuple[int, int, int] | None = None
    if sys.platform == "darwin":
        actual_boundary_kind = "darwin_sandbox_no_descendants"
    elif sys.platform == "linux":
        pre_linux_status = _attest_native_linux_amd64()
        actual_boundary_kind = "linux_seccomp_amd64_no_descendants"
    else:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    if boundary_kind != actual_boundary_kind:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)

    if actual_boundary_kind == "darwin_sandbox_no_descendants":
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            if resource.getrlimit(resource.RLIMIT_NPROC) != (0, 0):
                raise RuntimeError
        except Exception:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
    else:
        _install_linux_seccomp_amd64()
    try:
        pid = os.fork()
    except OSError as exc:
        fork_errno = exc.errno
        pid = None
    if pid is None:
        import threading

        completed: list[bool] = []
        thread = threading.Thread(target=lambda: completed.append(True))
        thread.start()
        thread.join()
        if completed != [True]:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        if actual_boundary_kind == "darwin_sandbox_no_descendants":
            return (
                "darwin-no-descendants-v1",
                "darwin",
                platform.machine().lower(),
                "rlimit-nproc-zero",
                "fork-denied",
                "thread-ok",
            )
        if fork_errno != errno.EPERM or pre_linux_status is None:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        try:
            os.execve("/hosted-byok-exec-probe-does-not-exist", ["probe"], {})
        except OSError as exc:
            if exc.errno != errno.EPERM:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE) from None
        else:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        no_new_privs, seccomp_mode, seccomp_filters = _linux_status_values()
        libc = ctypes.CDLL(None, use_errno=True)
        if (
            libc.prctl(39, 0, 0, 0, 0) != 1  # PR_GET_NO_NEW_PRIVS
            or libc.prctl(21, 0, 0, 0, 0) != 2  # PR_GET_SECCOMP
            or no_new_privs != 1
            or seccomp_mode != 2
            or seccomp_filters != pre_linux_status[2] + 1
        ):
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        return (
            "linux-seccomp-amd64-v1",
            "linux",
            "x86_64",
            "elf64-x86-64",
            no_new_privs,
            seccomp_mode,
            "fork-eperm",
            "exec-eperm",
            "thread-ok",
            seccomp_filters,
        )
    if pid == 0:
        os._exit(91)
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    finally:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)


def _send_framed_blocking(peer: socket.socket, payload: bytes) -> None:
    if type(payload) is not bytes or len(payload) > _MAX_PROCESS_IPC_BYTES:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    peer.sendall(struct.pack("!I", len(payload)) + payload)


def _receive_framed_blocking(peer: socket.socket) -> bytes:
    def receive_exact(size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = peer.recv(size - len(result))
            if not chunk:
                raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
            result.extend(chunk)
        return bytes(result)

    size = struct.unpack("!I", receive_exact(4))[0]
    if size > _MAX_PROCESS_IPC_BYTES:
        raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
    return receive_exact(size)


def _hosted_byok_clean_child_main(
    connection_fd: int,
    challenge_hex: str,
    boundary_kind: str,
) -> None:
    """Clean-start child: attest containment before receiving the credential-bearing request."""

    connection = socket.socket(fileno=connection_fd)
    try:
        if type(challenge_hex) is not str:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        challenge = bytes.fromhex(challenge_hex)
        if len(challenge) != 32:
            raise HostedByokInvocationError(_PROVIDER_FAILURE_CODE)
        environment_attestation = _attest_clean_child_environment()
        interpreter_attestation = _current_executable_attestation()
        _scrub_provider_environment()
        containment_attestation = _establish_no_descendant_boundary(boundary_kind)
        _send_framed_blocking(
            connection,
            _wire_dumps(
                (
                    "ready",
                    HOSTED_BYOK_PROCESS_ISOLATION_VERSION,
                    challenge,
                    boundary_kind,
                    interpreter_attestation,
                    containment_attestation,
                    environment_attestation,
                )
            ),
        )
        worker_payload = _receive_framed_blocking(connection)

        async def run() -> Verdict:
            _, child_wire_loads, child_wire_value = _make_private_wire_codec()
            child_validate_inputs = _make_private_input_validator(child_wire_value)
            values = child_wire_loads(worker_payload)
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
            child_validate_inputs(
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
            _send_framed_blocking(connection, _wire_dumps(("ok", result)))
        except HostedByokDeadlineExceeded as exc:
            _send_framed_blocking(connection, _wire_dumps(("deadline", str(exc))))
        except HostedByokCancelled:
            _send_framed_blocking(connection, _wire_dumps(("cancelled", "")))
        except HostedByokUnsupportedError as exc:
            _send_framed_blocking(connection, _wire_dumps(("unsupported", str(exc))))
        except HostedByokContractError as exc:
            _send_framed_blocking(connection, _wire_dumps(("contract", str(exc))))
        except BaseException:
            _send_framed_blocking(
                connection,
                _wire_dumps(("failure", _PROVIDER_FAILURE_CODE)),
            )
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
