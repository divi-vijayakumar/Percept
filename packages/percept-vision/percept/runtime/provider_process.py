"""One-call exec boundary for effectful vision-provider roots.

The parent owns admission, cognition, and reduction.  A fresh interpreter owns
one provider call and everything it can create: direct ``Task`` objects,
callbacks, executor threads, and subprocess descendants.  The parent does not
decode or publish the result until it has killed the child's process group and
joined the root process.
"""
from __future__ import annotations

import asyncio
import base64
import errno
import importlib
import json
import math
import os
import platform
import re
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
import types
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeVar

_T = TypeVar("_T")

_READY = b"R"
_FRAME_HEADER = struct.Struct("!Q")
_MAX_REQUEST_BYTES = 256 * 1024 * 1024
_MAX_RESULT_BYTES = 4 * 1024 * 1024
_MAX_WIRE_DEPTH = 64
_MAX_WIRE_NODES = 1_000_000
_CHILD_ARGUMENT = "--percept-provider-child"
_DARWIN_SANDBOX_PROFILE = "(version 1)(allow default)(deny process-fork)"
_SCHEMA_VERSION = 1
_SDK_IMPORT_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
_ISOLATED_CHILD_BOOTSTRAP = (
    "import runpy,sys;"
    "sys.path.insert(0,sys.argv.pop(1));"
    "runpy.run_module('percept.runtime.provider_process',run_name='__main__')"
)
_CHILD_ENVIRONMENT = MappingProxyType(
    {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "TMPDIR": "/tmp",
        # macOS injects this locale selector at exec. Supplying its
        # deterministic, non-secret value keeps the child env exact.
        **(
            {"__CF_USER_TEXT_ENCODING": f"0x{os.getuid():X}:0x0:0x0"}
            if sys.platform == "darwin"
            else {}
        ),
    }
)
_OPERATIONS = frozenset(
    {"__call__", "judge", "detect", "detect_events", "summarize_process"}
)
_IMPORT_PART = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

_ACTIVE_PROVIDER_PROCESSES_LOCK = threading.Lock()
_ACTIVE_PROVIDER_PROCESSES: set[int] = set()


class ProviderProcessError(RuntimeError):
    """The external provider boundary failed before returning a frozen result."""


class ProviderWatchdogExpired(ProviderProcessError):
    """The provider exceeded its production execution deadline."""


class ProviderChildError(ProviderProcessError):
    """The provider child raised before its result crossed the boundary."""

    def __init__(self, exception_type: str) -> None:
        self.exception_type = exception_type
        super().__init__(
            f"provider child failed with {exception_type or 'unknown exception'}"
        )


@dataclass(frozen=True)
class ProcessProviderRegistration:
    """Reviewed recipe for recreating one exact custom adapter in the child."""

    module: str
    qualname: str
    constructor_fields: tuple[str, ...]


_PROCESS_PROVIDER_REGISTRATIONS: dict[
    type[Any], ProcessProviderRegistration
] = {}
_PROCESS_PROVIDER_REGISTRATIONS_LOCK = threading.Lock()


def _valid_import_name(value: str, *, dotted: bool) -> bool:
    if type(value) is not str or not value or len(value) > 512:
        return False
    parts = value.split(".") if dotted else (value,)
    return all(_IMPORT_PART.fullmatch(part) is not None for part in parts)


def register_process_provider(
    adapter_type: type[Any],
    *,
    module: str,
    qualname: str,
    constructor_fields: tuple[str, ...] = (),
) -> None:
    """Authorize a fixed import + exact primitive-field constructor contract.

    Registration is intentionally explicit: arbitrary adapter instances are
    never pickled, inspected through their hooks, or imported by class name.
    """
    if type(adapter_type) is not type:
        raise TypeError("process provider registration requires an exact class")
    if not _valid_import_name(module, dotted=True):
        raise ValueError("process provider module must be a dotted import name")
    if not _valid_import_name(qualname, dotted=True):
        raise ValueError("process provider qualname must be a dotted class name")
    if type(constructor_fields) is not tuple or any(
        type(field) is not str
        or _IMPORT_PART.fullmatch(field) is None
        for field in constructor_fields
    ):
        raise ValueError(
            "process provider constructor fields must be exact identifiers"
        )
    if len(set(constructor_fields)) != len(constructor_fields):
        raise ValueError("process provider constructor fields must be unique")
    class_namespace = type.__getattribute__(adapter_type, "__dict__")
    if constructor_fields and type(class_namespace.get("__dict__")) is not (
        types.GetSetDescriptorType
    ):
        raise ValueError(
            "registered provider constructor fields require a plain "
            "instance dictionary"
        )
    registration = ProcessProviderRegistration(
        module=module,
        qualname=qualname,
        constructor_fields=constructor_fields,
    )
    with _PROCESS_PROVIDER_REGISTRATIONS_LOCK:
        existing = _PROCESS_PROVIDER_REGISTRATIONS.get(adapter_type)
        if existing is not None and existing != registration:
            raise ValueError(
                "process provider class already has a different registration"
            )
        _PROCESS_PROVIDER_REGISTRATIONS[adapter_type] = registration


def _provider_child_command() -> list[str]:
    command = [
        sys.executable,
        "-I",
        "-c",
        _ISOLATED_CHILD_BOOTSTRAP,
        _SDK_IMPORT_ROOT,
        _CHILD_ARGUMENT,
    ]
    if sys.platform == "darwin":
        sandbox = "/usr/bin/sandbox-exec"
        try:
            metadata = os.stat(sandbox, follow_symlinks=False)
        except OSError as exc:
            raise ProviderProcessError(
                "trusted provider process containment is unavailable"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise ProviderProcessError(
                "trusted provider process containment failed validation"
            )
        return [sandbox, "-p", _DARWIN_SANDBOX_PROFILE, *command]
    if (
        sys.platform == "linux"
        and platform.machine().lower() in {"x86_64", "amd64"}
    ):
        return command
    raise ProviderProcessError(
        "trusted provider process containment is unavailable on this platform"
    )


def _provider_child_environment() -> dict[str, str]:
    """Return the fixed interpreter environment; never copy ambient values."""
    return dict(_CHILD_ENVIRONMENT)


def _install_linux_no_descendants_filter() -> None:
    """Deny process creation/exec while retaining CLONE_THREAD."""
    import ctypes

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
        (0x20, 0, 0, 4),
        (0x15, 1, 0, audit_arch_x86_64),
        (0x06, 0, 0, seccomp_ret_kill_process),
        (0x20, 0, 0, 0),
        (0x35, 0, 1, 0x40000000),
        (0x06, 0, 0, seccomp_ret_errno | errno.ENOSYS),
        (0x15, 0, 5, 56),
        (0x20, 0, 0, 16),
        (0x54, 0, 0, clone_thread),
        (0x15, 0, 1, 0),
        (0x06, 0, 0, seccomp_ret_errno | errno.EPERM),
        (0x06, 0, 0, seccomp_ret_allow),
        (0x15, 0, 1, 57),
        (0x06, 0, 0, seccomp_ret_errno | errno.EPERM),
        (0x15, 0, 1, 58),
        (0x06, 0, 0, seccomp_ret_errno | errno.EPERM),
        (0x15, 0, 1, 435),
        (0x06, 0, 0, seccomp_ret_errno | errno.ENOSYS),
        (0x15, 0, 1, 59),
        (0x06, 0, 0, seccomp_ret_errno | errno.EPERM),
        (0x15, 0, 1, 322),
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
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    libc.syscall.restype = ctypes.c_long
    if libc.syscall(317, 1, 1, ctypes.byref(program)) != 0:
        raise OSError(ctypes.get_errno(), "seccomp filter install failed")


def _establish_child_containment() -> None:
    if sys.platform == "darwin":
        # The parent launched this interpreter under the validated sandbox.
        return
    if (
        sys.platform == "linux"
        and platform.machine().lower() in {"x86_64", "amd64"}
    ):
        _install_linux_no_descendants_filter()
        return
    raise RuntimeError("provider child containment unavailable")


class _MalformedProviderResult:
    def __init__(self, exception_type: str) -> None:
        self.exception_type = exception_type


class _WireBudget:
    __slots__ = ("nodes",)

    def __init__(self) -> None:
        self.nodes = 0

    def take(self, depth: int) -> None:
        self.nodes += 1
        if depth > _MAX_WIRE_DEPTH or self.nodes > _MAX_WIRE_NODES:
            raise ValueError("provider wire graph exceeded its structural limit")


def _field(value: Any, name: str) -> Any:
    """Read a field without invoking an instance-defined attribute hook."""
    return object.__getattribute__(value, name)


def _finite_number(value: Any) -> int | float:
    if type(value) is int:
        if not -(2**63) <= value < 2**63:
            raise ValueError("provider integer exceeded the wire range")
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("provider value requires a finite number")


def _encode_plain_value(
    value: Any,
    budget: _WireBudget,
    depth: int = 0,
) -> Any:
    """Encode only exact primitives and reviewed SDK DTOs."""
    budget.take(depth)
    if value is None or type(value) in (bool, str):
        return value
    if type(value) in (int, float):
        return _finite_number(value)
    if type(value) in (bytes, bytearray):
        return {
            "$": "bytes",
            "v": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if type(value) is list:
        return {
            "$": "list",
            "v": [
                _encode_plain_value(item, budget, depth + 1)
                for item in value
            ],
        }
    if type(value) is tuple:
        return {
            "$": "tuple",
            "v": [
                _encode_plain_value(item, budget, depth + 1)
                for item in value
            ],
        }
    if type(value) in (set, frozenset):
        encoded = [
            _encode_plain_value(item, budget, depth + 1)
            for item in value
        ]
        encoded.sort(key=_canonical_json)
        return {
            "$": "frozenset" if type(value) is frozenset else "set",
            "v": encoded,
        }
    if type(value) in (dict, MappingProxyType):
        entries: list[list[Any]] = []
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(
                    "provider mapping keys must be exact strings"
                )
            entries.append(
                [key, _encode_plain_value(item, budget, depth + 1)]
            )
        entries.sort(key=lambda pair: pair[0])
        return {"$": "dict", "v": entries}

    from percept.contracts import (
        AudioFrame,
        Box,
        ClipEvent,
        Goal,
        GoalOrigin,
        SubjectRef,
    )
    from percept.harness.tier0 import Frame

    if type(value) is Box:
        return {
            "$": "box",
            "v": [
                _finite_number(_field(value, name))
                for name in ("x", "y", "w", "h")
            ],
        }
    if type(value) is Frame:
        width = _field(value, "w")
        height = _field(value, "h")
        pixels = _field(value, "pixels")
        if type(width) is not int or type(height) is not int:
            raise ValueError("provider Frame dimensions must be exact integers")
        return {
            "$": "frame",
            "w": _finite_number(width),
            "h": _finite_number(height),
            "pixels": _encode_plain_value(pixels, budget, depth + 1),
        }
    if type(value) is AudioFrame:
        sample_rate = _field(value, "sample_rate")
        channels = _field(value, "channels")
        sample_format = _field(value, "sample_format")
        mime = _field(value, "mime")
        if (
            type(sample_rate) is not int
            or type(channels) is not int
            or type(sample_format) is not str
            or (mime is not None and type(mime) is not str)
        ):
            raise ValueError("provider AudioFrame field is malformed")
        return {
            "$": "audio_frame",
            "pcm": _encode_plain_value(
                _field(value, "pcm"), budget, depth + 1
            ),
            "sample_rate": _finite_number(sample_rate),
            "channels": _finite_number(channels),
            "sample_format": sample_format,
            "encoded": _encode_plain_value(
                _field(value, "encoded"), budget, depth + 1
            ),
            "mime": mime,
            "t_start": _encode_plain_value(
                _field(value, "t_start"), budget, depth + 1
            ),
            "t_end": _encode_plain_value(
                _field(value, "t_end"), budget, depth + 1
            ),
        }
    if type(value) is Goal:
        origin = _field(value, "origin")
        if origin is GoalOrigin.STRUCTURED_TRUSTED:
            encoded_origin = "structured_trusted"
        elif origin is GoalOrigin.LANGUAGE_AUTHORED:
            encoded_origin = "language_authored"
        elif origin is GoalOrigin.RESTORED:
            encoded_origin = "restored"
        elif origin is GoalOrigin.SYSTEM_TEMPLATE:
            encoded_origin = "system_template"
        else:
            raise ValueError("provider Goal has an invalid origin")
        return {
            "$": "goal",
            "id": _encode_exact_string(_field(value, "id")),
            "condition": _encode_exact_string(_field(value, "condition")),
            "say": _encode_exact_string(_field(value, "say")),
            "source": _encode_exact_string(_field(value, "source")),
            "modality": _encode_exact_string(_field(value, "modality")),
            "state": _encode_plain_value(
                _field(value, "state"), budget, depth + 1
            ),
            "expiry": _encode_exact_string(_field(value, "expiry")),
            "origin": encoded_origin,
        }
    if type(value) is ClipEvent:
        return {
            "$": "clip_event",
            "t_start": _finite_number(_field(value, "t_start")),
            "t_end": _encode_plain_value(
                _field(value, "t_end"), budget, depth + 1
            ),
            "label": _encode_exact_string(_field(value, "label")),
        }
    if type(value) is SubjectRef:
        return {
            "$": "subject_ref",
            "scheme": _encode_exact_string(_field(value, "scheme")),
            "namespace": _encode_exact_string(_field(value, "namespace")),
            "key": _encode_optional_string(_field(value, "key")),
            "resolution": _encode_exact_string(
                _field(value, "resolution")
            ),
            "scope": _encode_exact_string(_field(value, "scope")),
            "strategy": _encode_exact_string(_field(value, "strategy")),
            "confidence": _encode_plain_value(
                _field(value, "confidence"), budget, depth + 1
            ),
        }
    raise ValueError("provider value is not in the reviewed wire schema")


def _encode_exact_string(value: Any) -> str:
    if type(value) is not str:
        raise ValueError("provider DTO string field must be an exact string")
    return value


def _encode_optional_string(value: Any) -> str | None:
    if value is None or type(value) is str:
        return value
    raise ValueError("provider DTO optional string field is malformed")


def _decode_plain_value(
    value: Any,
    budget: _WireBudget,
    depth: int = 0,
) -> Any:
    budget.take(depth)
    if value is None or type(value) in (bool, str):
        return value
    if type(value) in (int, float):
        return _finite_number(value)
    if type(value) is not dict or type(value.get("$")) is not str:
        raise ValueError("provider wire value has an invalid shape")
    tag = value["$"]
    if tag == "bytes" and set(value) == {"$", "v"}:
        encoded = value["v"]
        if type(encoded) is not str:
            raise ValueError("provider bytes value is malformed")
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("provider bytes value is malformed") from exc
    if tag in ("list", "tuple", "set", "frozenset") and set(value) == {
        "$",
        "v",
    }:
        items = value["v"]
        if type(items) is not list:
            raise ValueError("provider collection value is malformed")
        decoded = [
            _decode_plain_value(item, budget, depth + 1)
            for item in items
        ]
        if tag == "list":
            return decoded
        if tag == "tuple":
            return tuple(decoded)
        try:
            return frozenset(decoded) if tag == "frozenset" else set(decoded)
        except TypeError as exc:
            raise ValueError(
                "provider set contains an unhashable value"
            ) from exc
    if tag == "dict" and set(value) == {"$", "v"}:
        entries = value["v"]
        if type(entries) is not list:
            raise ValueError("provider mapping value is malformed")
        result: dict[str, Any] = {}
        for pair in entries:
            if (
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0]) is not str
                or pair[0] in result
            ):
                raise ValueError("provider mapping entry is malformed")
            result[pair[0]] = _decode_plain_value(
                pair[1], budget, depth + 1
            )
        return result

    from percept.contracts import (
        AudioFrame,
        Box,
        ClipEvent,
        Goal,
        GoalOrigin,
        SubjectRef,
    )
    from percept.harness.tier0 import Frame

    if tag == "box" and set(value) == {"$", "v"}:
        fields = value["v"]
        if type(fields) is not list or len(fields) != 4:
            raise ValueError("provider Box value is malformed")
        return Box(*(_finite_number(item) for item in fields))
    if tag == "frame" and set(value) == {"$", "w", "h", "pixels"}:
        width = value["w"]
        height = value["h"]
        pixels = _decode_plain_value(
            value["pixels"], budget, depth + 1
        )
        if (
            type(width) is not int
            or type(height) is not int
            or type(pixels) is not tuple
            or any(type(pixel) is not int for pixel in pixels)
        ):
            raise ValueError("provider Frame value is malformed")
        return Frame(w=width, h=height, pixels=pixels)
    if tag == "audio_frame" and set(value) == {
        "$",
        "pcm",
        "sample_rate",
        "channels",
        "sample_format",
        "encoded",
        "mime",
        "t_start",
        "t_end",
    }:
        decoded = {
            key: _decode_plain_value(value[key], budget, depth + 1)
            for key in ("pcm", "encoded", "t_start", "t_end")
        }
        if (
            type(value["sample_rate"]) is not int
            or type(value["channels"]) is not int
            or type(value["sample_format"]) is not str
            or (value["mime"] is not None and type(value["mime"]) is not str)
        ):
            raise ValueError("provider AudioFrame value is malformed")
        return AudioFrame(
            pcm=decoded["pcm"],
            sample_rate=value["sample_rate"],
            channels=value["channels"],
            sample_format=value["sample_format"],
            encoded=decoded["encoded"],
            mime=value["mime"],
            t_start=decoded["t_start"],
            t_end=decoded["t_end"],
        )
    if tag == "goal" and set(value) == {
        "$",
        "id",
        "condition",
        "say",
        "source",
        "modality",
        "state",
        "expiry",
        "origin",
    }:
        string_keys = (
            "id",
            "condition",
            "say",
            "source",
            "modality",
            "expiry",
            "origin",
        )
        if any(type(value[key]) is not str for key in string_keys):
            raise ValueError("provider Goal value is malformed")
        origins = {origin.value: origin for origin in GoalOrigin}
        if value["origin"] not in origins:
            raise ValueError("provider Goal origin is malformed")
        state = _decode_plain_value(value["state"], budget, depth + 1)
        if type(state) is not dict:
            raise ValueError("provider Goal state is malformed")
        return Goal(
            id=value["id"],
            condition=value["condition"],
            say=value["say"],
            source=value["source"],
            modality=value["modality"],
            state=state,
            expiry=value["expiry"],
            origin=origins[value["origin"]],
        )
    if tag == "clip_event" and set(value) == {
        "$",
        "t_start",
        "t_end",
        "label",
    }:
        t_end = _decode_plain_value(value["t_end"], budget, depth + 1)
        if type(value["label"]) is not str:
            raise ValueError("provider ClipEvent value is malformed")
        return ClipEvent(
            t_start=_finite_number(value["t_start"]),
            t_end=t_end,
            label=value["label"],
        )
    if tag == "subject_ref" and set(value) == {
        "$",
        "scheme",
        "namespace",
        "key",
        "resolution",
        "scope",
        "strategy",
        "confidence",
    }:
        if any(
            type(value[key]) is not str
            for key in (
                "scheme",
                "namespace",
                "resolution",
                "scope",
                "strategy",
            )
        ) or (value["key"] is not None and type(value["key"]) is not str):
            raise ValueError("provider SubjectRef value is malformed")
        confidence = _decode_plain_value(
            value["confidence"], budget, depth + 1
        )
        return SubjectRef(
            scheme=value["scheme"],
            namespace=value["namespace"],
            key=value["key"],
            resolution=value["resolution"],
            scope=value["scope"],
            strategy=value["strategy"],
            confidence=confidence,
        )
    raise ValueError("provider wire value has an unsupported tag")


def _encode_provider_result(value: Any) -> Any:
    """Convert provider output to an exact, non-executable JSON graph."""
    from percept.contracts import Observation, Verdict
    from percept.events import Event

    budget = _WireBudget()
    if type(value) is Verdict:
        fields = {
            name: _field(value, name)
            for name in (
                "satisfied",
                "confidence",
                "evidence",
                "phase",
                "action_present",
                "running_answer",
                "usage",
                "events",
                "subject",
                "operation",
                "value",
                "normalized_value",
                "target_revision_id",
                "asserted_by",
            )
        }
        if (
            type(fields["satisfied"]) is not bool
            or type(fields["confidence"]) not in (int, float)
            or type(fields["evidence"]) is not str
            or type(fields["phase"]) is not str
            or fields["action_present"] is not None
            and type(fields["action_present"]) is not bool
            or type(fields["running_answer"]) is not str
            or type(fields["operation"]) is not str
            or (
                fields["target_revision_id"] is not None
                and type(fields["target_revision_id"]) is not str
            )
            or (
                fields["asserted_by"] is not None
                and type(fields["asserted_by"]) is not str
            )
        ):
            return {"$": "malformed", "reason": "malformed Verdict"}
        try:
            encoded_fields = {
                name: _encode_plain_value(item, budget)
                for name, item in fields.items()
            }
        except ValueError:
            return {"$": "malformed", "reason": "malformed Verdict"}
        return {"$": "verdict", "v": encoded_fields}
    if type(value) is Observation:
        try:
            return _encode_observation(value, budget)
        except ValueError:
            return {"$": "malformed", "reason": "malformed Observation"}
    if type(value) is Event:
        try:
            return _encode_event(value, budget)
        except ValueError:
            return {"$": "malformed", "reason": "malformed Event"}
    if type(value) in (list, tuple):
        try:
            items = [
                (
                    _encode_observation(item, budget)
                    if type(item) is Observation
                    else _encode_event(item, budget)
                    if type(item) is Event
                    else _encode_plain_value(item, budget)
                )
                for item in value
            ]
        except ValueError:
            return {"$": "malformed", "reason": "unsupported completion"}
        return {
            "$": "result_tuple" if type(value) is tuple else "result_list",
            "v": items,
        }
    try:
        return {"$": "plain", "v": _encode_plain_value(value, budget)}
    except ValueError:
        return {"$": "malformed", "reason": "unsupported completion"}


def _encode_observation(value: Any, budget: _WireBudget) -> dict[str, Any]:
    from percept.contracts import Box

    label = _field(value, "label")
    box = _field(value, "box")
    confidence = _field(value, "confidence")
    attrs = _field(value, "attrs")
    if (
        type(label) is not str
        or type(box) is not Box
        or type(confidence) not in (int, float)
        or type(attrs) is not dict
    ):
        raise ValueError("malformed Observation")
    return {
        "$": "observation",
        "label": label,
        "box": _encode_plain_value(box, budget),
        "confidence": _finite_number(confidence),
        "attrs": _encode_plain_value(attrs, budget),
    }


def _encode_event(value: Any, budget: _WireBudget) -> dict[str, Any]:
    event_type = _field(value, "type")
    timestamp = _field(value, "t")
    participants = _field(value, "participants")
    confidence = _field(value, "confidence")
    event_id = _field(value, "id")
    if (
        type(event_type) is not str
        or type(timestamp) not in (int, float)
        or type(participants) is not list
        or type(confidence) not in (int, float)
        or (event_id is not None and type(event_id) is not str)
    ):
        raise ValueError("malformed Event")
    return {
        "$": "event",
        "type": event_type,
        "t": _finite_number(timestamp),
        "participants": _encode_plain_value(participants, budget),
        "confidence": _finite_number(confidence),
        "id": event_id,
    }


def _decode_provider_value(value: Any) -> Any:
    if type(value) is not dict or type(value.get("$")) is not str:
        raise ProviderProcessError(
            "provider process returned an unsupported completion"
        )
    tag = value["$"]
    if tag == "malformed" and set(value) == {"$", "reason"}:
        if type(value["reason"]) is not str:
            raise ProviderProcessError(
                "provider process returned a malformed failure"
            )
        return _MalformedProviderResult(value["reason"])
    if tag == "plain" and set(value) == {"$", "v"}:
        try:
            return _decode_plain_value(value["v"], _WireBudget())
        except ValueError as exc:
            raise ProviderProcessError(
                "provider process returned a malformed plain completion"
            ) from exc
    if tag in ("result_list", "result_tuple") and set(value) == {"$", "v"}:
        if type(value["v"]) is not list:
            raise ProviderProcessError(
                "provider process returned a malformed collection"
            )
        items = [_decode_result_item(item) for item in value["v"]]
        return tuple(items) if tag == "result_tuple" else items
    if tag == "verdict" and set(value) == {"$", "v"}:
        fields = value["v"]
        expected = {
            "satisfied",
            "confidence",
            "evidence",
            "phase",
            "action_present",
            "running_answer",
            "usage",
            "events",
            "subject",
            "operation",
            "value",
            "normalized_value",
            "target_revision_id",
            "asserted_by",
        }
        if type(fields) is not dict or set(fields) != expected:
            raise ProviderProcessError(
                "provider process returned a malformed Verdict"
            )
        try:
            decoded = {
                name: _decode_plain_value(item, _WireBudget())
                for name, item in fields.items()
            }
        except ValueError as exc:
            raise ProviderProcessError(
                "provider process returned a malformed Verdict"
            ) from exc
        from percept.contracts import Verdict

        if (
            type(decoded["satisfied"]) is not bool
            or type(decoded["confidence"]) not in (int, float)
            or type(decoded["evidence"]) is not str
            or type(decoded["phase"]) is not str
            or (
                decoded["action_present"] is not None
                and type(decoded["action_present"]) is not bool
            )
            or type(decoded["running_answer"]) is not str
            or type(decoded["usage"]) is not dict
            or type(decoded["events"]) is not list
            or type(decoded["operation"]) is not str
            or (
                decoded["target_revision_id"] is not None
                and type(decoded["target_revision_id"]) is not str
            )
            or (
                decoded["asserted_by"] is not None
                and type(decoded["asserted_by"]) is not str
            )
        ):
            raise ProviderProcessError(
                "provider process returned a malformed Verdict"
            )
        return Verdict(**decoded)
    return _decode_result_item(value)


def _decode_result_item(value: Any) -> Any:
    if type(value) is not dict or type(value.get("$")) is not str:
        try:
            return _decode_plain_value(value, _WireBudget())
        except ValueError as exc:
            raise ProviderProcessError(
                "provider process returned a malformed result item"
            ) from exc
    tag = value["$"]
    if tag == "observation" and set(value) == {
        "$",
        "label",
        "box",
        "confidence",
        "attrs",
    }:
        try:
            box = _decode_plain_value(value["box"], _WireBudget())
            attrs = _decode_plain_value(value["attrs"], _WireBudget())
        except ValueError as exc:
            raise ProviderProcessError(
                "provider process returned a malformed Observation"
            ) from exc
        from percept.contracts import Box, Observation

        if (
            type(value["label"]) is not str
            or type(box) is not Box
            or type(value["confidence"]) not in (int, float)
            or type(attrs) is not dict
        ):
            raise ProviderProcessError(
                "provider process returned a malformed Observation"
            )
        return Observation(
            label=value["label"],
            box=box,
            confidence=_finite_number(value["confidence"]),
            attrs=attrs,
        )
    if tag == "event" and set(value) == {
        "$",
        "type",
        "t",
        "participants",
        "confidence",
        "id",
    }:
        try:
            participants = _decode_plain_value(
                value["participants"], _WireBudget()
            )
        except ValueError as exc:
            raise ProviderProcessError(
                "provider process returned a malformed Event"
            ) from exc
        if (
            type(value["type"]) is not str
            or type(value["t"]) not in (int, float)
            or type(participants) is not list
            or type(value["confidence"]) not in (int, float)
            or (value["id"] is not None and type(value["id"]) is not str)
        ):
            raise ProviderProcessError(
                "provider process returned a malformed Event"
            )
        from percept.events import Event

        return Event(
            type=value["type"],
            t=_finite_number(value["t"]),
            participants=participants,
            confidence=_finite_number(value["confidence"]),
            id=value["id"],
        )
    try:
        return _decode_plain_value(value, _WireBudget())
    except ValueError as exc:
        raise ProviderProcessError(
            "provider process returned an unsupported completion"
        ) from exc


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _encode_json_bytes(value: Any, *, limit: int) -> bytes:
    try:
        payload = _canonical_json(value).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProviderProcessError(
            "provider value is outside the wire schema"
        ) from exc
    if len(payload) > limit:
        raise ProviderProcessError(
            "provider frame exceeded the process boundary limit"
        )
    return payload


def _decode_json_bytes(payload: bytes) -> Any:
    try:
        text = payload.decode("ascii")
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProviderProcessError(
            "provider process returned malformed JSON"
        ) from exc


def _decode_provider_result(payload: bytes) -> Any:
    envelope = _decode_json_bytes(payload)
    if (
        type(envelope) is not dict
        or set(envelope) != {"schema", "status", "value"}
        or envelope["schema"] != _SCHEMA_VERSION
        or envelope["status"] not in ("ok", "error")
    ):
        raise ProviderProcessError(
            "provider process returned an invalid completion envelope"
        )
    if envelope["status"] == "error":
        if type(envelope["value"]) is not str:
            raise ProviderProcessError(
                "provider process returned an invalid error envelope"
            )
        raise ProviderChildError(envelope["value"])
    return _decode_provider_value(envelope["value"])


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise BrokenPipeError("provider process channel closed during write")
        view = view[written:]


def _read_exact_fd(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(fd, remaining)
        if not chunk:
            raise EOFError("provider process channel closed during read")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame_fd(fd: int, *, limit: int) -> bytes:
    header = _read_exact_fd(fd, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    if size > limit:
        raise ValueError("provider process frame exceeded its boundary limit")
    return _read_exact_fd(fd, size)


def _send_frame_fd(fd: int, payload: bytes, *, limit: int) -> None:
    if len(payload) > limit:
        raise ValueError("provider process frame exceeded its boundary limit")
    _write_all(fd, _FRAME_HEADER.pack(len(payload)) + payload)


def _adapter_state(adapter: Any) -> dict[str, Any]:
    try:
        state = object.__getattribute__(adapter, "__dict__")
    except BaseException as exc:  # noqa: BLE001 - never consult provider hooks
        raise ProviderProcessError(
            "provider adapter has no reviewed constructor state"
        ) from exc
    if type(state) is not dict:
        raise ProviderProcessError(
            "provider adapter has no reviewed constructor state"
        )
    return state


def _reviewed_builtin_credential(
    state: dict[str, Any],
    aliases: tuple[str, ...],
) -> str | None:
    """Select one exact credential for the typed recipe, never child env."""
    explicit = state.get("_api_key")
    if explicit is not None:
        if type(explicit) is not str or not explicit or len(explicit) > 65_536:
            raise ProviderProcessError(
                "explicit provider credential is outside the reviewed schema"
            )
        return explicit
    for alias in aliases:
        credential = os.environ.get(alias)
        if credential:
            if type(credential) is not str or len(credential) > 65_536:
                raise ProviderProcessError(
                    "provider credential is outside the reviewed schema"
                )
            return credential
    return None


def _adapter_spec(
    adapter: Any,
    *,
    seen: set[int] | None = None,
) -> dict[str, Any]:
    """Resolve an exact registered recipe without invoking provider code."""
    if adapter is None:
        raise ProviderProcessError(
            "process provider dispatch requires an adapter instance"
        )
    if seen is None:
        seen = set()
    identity = id(adapter)
    if identity in seen:
        raise ProviderProcessError("provider adapter graph contains a cycle")
    seen.add(identity)
    try:
        adapter_type = type(adapter)
        if type(adapter_type) is not type:
            raise ProviderProcessError(
                "provider adapter class requires the standard metaclass"
            )
        from percept.backends.anthropic import AnthropicVision
        from percept.backends.fallback import FallbackVision
        from percept.backends.gemini import GeminiVision
        from percept.backends.gemini_live import GeminiLiveVision
        from percept.backends.openrouter import OpenRouterVision

        builtins = {
            AnthropicVision: (
                "anthropic",
                ("ANTHROPIC_API_KEY",),
            ),
            GeminiVision: (
                "gemini",
                (
                    "GEMINI_API_KEY",
                    "GOOGLE_API_KEY",
                    "GOOGLE_AISTUDIO_API_KEY",
                ),
            ),
            GeminiLiveVision: (
                "gemini_live",
                (
                    "GEMINI_API_KEY",
                    "GOOGLE_API_KEY",
                    "GOOGLE_AISTUDIO_API_KEY",
                ),
            ),
            OpenRouterVision: (
                "openrouter",
                ("OPENROUTER_API_KEY", "OPENROUTER_KEY"),
            ),
        }
        builtin_recipe = builtins.get(adapter_type)
        if builtin_recipe is not None:
            builtin_id, credential_aliases = builtin_recipe
            state = _adapter_state(adapter)
            if state.get("_client") is not None:
                raise ProviderProcessError(
                    "injected provider clients cannot cross the process boundary"
                )
            if (
                builtin_id == "openrouter"
                and state.get("_request_budget") is not None
            ):
                raise ProviderProcessError(
                    "request-budget callbacks cannot cross the process boundary"
                )
            model = state.get("_model")
            if type(model) is not str:
                raise ProviderProcessError(
                    "provider adapter model must be an exact string"
                )
            credential = _reviewed_builtin_credential(
                state,
                credential_aliases,
            )
            if credential is None:
                raise ProviderProcessError(
                    "builtin process provider requires an explicit "
                    "reviewed credential"
                )
            return {
                "kind": "builtin",
                "id": builtin_id,
                "model": model,
                "credential": credential,
            }
        if adapter_type is FallbackVision:
            state = _adapter_state(adapter)
            primary = state.get("_primary")
            fallback = state.get("_fallback")
            return {
                "kind": "fallback",
                "primary": _adapter_spec(primary, seen=seen),
                "fallback": (
                    None
                    if fallback is None
                    else _adapter_spec(fallback, seen=seen)
                ),
            }
        with _PROCESS_PROVIDER_REGISTRATIONS_LOCK:
            registration = _PROCESS_PROVIDER_REGISTRATIONS.get(adapter_type)
        if registration is None:
            raise ProviderProcessError(
                "custom process provider is not explicitly registered"
            )
        state = (
            _adapter_state(adapter)
            if registration.constructor_fields
            else {}
        )
        constructor: dict[str, Any] = {}
        budget = _WireBudget()
        for name in registration.constructor_fields:
            if name not in state:
                raise ProviderProcessError(
                    "registered provider constructor field is missing"
                )
            try:
                constructor[name] = _encode_plain_value(
                    state[name], budget
                )
            except ValueError as exc:
                raise ProviderProcessError(
                    "registered provider constructor field is outside "
                    "the reviewed wire schema"
                ) from exc
        return {
            "kind": "registered",
            "module": registration.module,
            "qualname": registration.qualname,
            "constructor": constructor,
        }
    finally:
        seen.discard(identity)


def _build_request(
    adapter: Any,
    operation: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bytes:
    if type(operation) is not str or operation not in _OPERATIONS:
        raise ProviderProcessError(
            "provider operation is outside the reviewed dispatch set"
        )
    if type(args) is not tuple or type(kwargs) is not dict:
        raise ProviderProcessError(
            "provider call arguments require exact tuple/dict containers"
        )
    if any(type(key) is not str for key in kwargs):
        raise ProviderProcessError(
            "provider keyword names must be exact strings"
        )
    budget = _WireBudget()
    try:
        encoded_args = [
            _encode_plain_value(item, budget) for item in args
        ]
        encoded_kwargs = {
            key: _encode_plain_value(kwargs[key], budget)
            for key in sorted(kwargs)
        }
    except ValueError as exc:
        raise ProviderProcessError(
            "provider call arguments are outside the reviewed wire schema"
        ) from exc
    return _encode_json_bytes(
        {
            "schema": _SCHEMA_VERSION,
            "provider": _adapter_spec(adapter),
            "operation": operation,
            "args": encoded_args,
            "kwargs": encoded_kwargs,
        },
        limit=_MAX_REQUEST_BYTES,
    )


def _decode_request(
    payload: bytes,
) -> tuple[dict[str, Any], str, tuple[Any, ...], dict[str, Any]]:
    request = _decode_json_bytes(payload)
    if (
        type(request) is not dict
        or set(request) != {
            "schema",
            "provider",
            "operation",
            "args",
            "kwargs",
        }
        or request["schema"] != _SCHEMA_VERSION
        or type(request["provider"]) is not dict
        or type(request["operation"]) is not str
        or request["operation"] not in _OPERATIONS
        or type(request["args"]) is not list
        or type(request["kwargs"]) is not dict
        or any(type(key) is not str for key in request["kwargs"])
    ):
        raise ProviderProcessError("provider request envelope is malformed")
    budget = _WireBudget()
    try:
        args = tuple(
            _decode_plain_value(item, budget) for item in request["args"]
        )
        kwargs = {
            key: _decode_plain_value(item, budget)
            for key, item in request["kwargs"].items()
        }
    except ValueError as exc:
        raise ProviderProcessError(
            "provider request arguments are malformed"
        ) from exc
    return request["provider"], request["operation"], args, kwargs


def _construct_adapter(spec: dict[str, Any]) -> Any:
    if type(spec) is not dict or type(spec.get("kind")) is not str:
        raise ProviderProcessError("provider reconstruction recipe is malformed")
    kind = spec["kind"]
    if kind == "builtin" and set(spec) == {
        "kind",
        "id",
        "model",
        "credential",
    }:
        if (
            type(spec["id"]) is not str
            or type(spec["model"]) is not str
            or type(spec["credential"]) is not str
            or not spec["credential"]
            or len(spec["credential"]) > 65_536
        ):
            raise ProviderProcessError(
                "builtin provider reconstruction recipe is malformed"
            )
        from percept.config import ModelSettings

        settings = ModelSettings(vision_model=spec["model"])
        if spec["id"] == "anthropic":
            from percept.backends.anthropic import AnthropicVision

            return AnthropicVision(settings, api_key=spec["credential"])
        if spec["id"] == "gemini":
            from percept.backends.gemini import GeminiVision

            return GeminiVision(settings, api_key=spec["credential"])
        if spec["id"] == "gemini_live":
            from percept.backends.gemini_live import GeminiLiveVision

            return GeminiLiveVision(settings, api_key=spec["credential"])
        if spec["id"] == "openrouter":
            from percept.backends.openrouter import OpenRouterVision

            return OpenRouterVision(settings, api_key=spec["credential"])
        raise ProviderProcessError("builtin provider id is not reviewed")
    if kind == "fallback" and set(spec) == {
        "kind",
        "primary",
        "fallback",
    }:
        if type(spec["primary"]) is not dict or (
            spec["fallback"] is not None
            and type(spec["fallback"]) is not dict
        ):
            raise ProviderProcessError(
                "fallback provider reconstruction recipe is malformed"
            )
        from percept.backends.fallback import FallbackVision

        return FallbackVision(
            _construct_adapter(spec["primary"]),
            (
                None
                if spec["fallback"] is None
                else _construct_adapter(spec["fallback"])
            ),
        )
    if kind == "registered" and set(spec) == {
        "kind",
        "module",
        "qualname",
        "constructor",
    }:
        module_name = spec["module"]
        qualname = spec["qualname"]
        constructor = spec["constructor"]
        if (
            not _valid_import_name(module_name, dotted=True)
            or not _valid_import_name(qualname, dotted=True)
            or type(constructor) is not dict
            or any(
                type(name) is not str
                or _IMPORT_PART.fullmatch(name) is None
                for name in constructor
            )
        ):
            raise ProviderProcessError(
                "registered provider reconstruction recipe is malformed"
            )
        module = importlib.import_module(module_name)
        target: Any = module
        for part in qualname.split("."):
            namespace = vars(target)
            if part not in namespace:
                raise ProviderProcessError(
                    "registered provider class is not importable"
                )
            target = namespace[part]
        if type(target) is not type:
            raise ProviderProcessError(
                "registered provider target is not an exact class"
            )
        budget = _WireBudget()
        try:
            kwargs = {
                name: _decode_plain_value(value, budget)
                for name, value in constructor.items()
            }
        except ValueError as exc:
            raise ProviderProcessError(
                "registered provider constructor is malformed"
            ) from exc
        return target(**kwargs)
    raise ProviderProcessError(
        "provider reconstruction recipe is outside the reviewed schema"
    )


def _child_entry() -> None:
    """Read one typed call, reconstruct it in containment, and await death."""
    stdout_fd = sys.stdout.fileno()
    try:
        _establish_child_containment()
        _write_all(stdout_fd, _READY)
        payload = _read_frame_fd(sys.stdin.fileno(), limit=_MAX_REQUEST_BYTES)
        provider_spec, operation, args, kwargs = _decode_request(payload)
        adapter = _construct_adapter(provider_spec)

        async def invoke() -> Any:
            dispatch = (
                adapter
                if operation == "__call__"
                else getattr(adapter, operation)
            )
            return await dispatch(*args, **kwargs)

        try:
            # Stop exactly when the provider root returns.  ``asyncio.run``
            # would perform cooperative task/executor shutdown and can wait
            # forever on resistant descendants.  The OS process boundary,
            # below, is the sole teardown authority.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            value = loop.run_until_complete(invoke())
            envelope: dict[str, Any] = {
                "schema": _SCHEMA_VERSION,
                "status": "ok",
                "value": _encode_provider_result(value),
            }
        except BaseException as exc:  # noqa: BLE001 - safe type-only wire error
            envelope = {
                "schema": _SCHEMA_VERSION,
                "status": "error",
                "value": type(exc).__name__,
            }
        encoded = _encode_json_bytes(envelope, limit=_MAX_RESULT_BYTES)
        _send_frame_fd(stdout_fd, encoded, limit=_MAX_RESULT_BYTES)
    except BaseException as exc:  # noqa: BLE001 - best-effort protocol failure
        try:
            encoded = _encode_json_bytes(
                {
                    "schema": _SCHEMA_VERSION,
                    "status": "error",
                    "value": type(exc).__name__,
                },
                limit=_MAX_RESULT_BYTES,
            )
            _send_frame_fd(stdout_fd, encoded, limit=_MAX_RESULT_BYTES)
        except BaseException:  # noqa: BLE001 - parent detects EOF
            pass
    finally:
        # Keep the process-group leader alive after publishing the result.
        # The parent enforces terminal quiescence by SIGKILLing this group and
        # joining the leader before it decodes or returns the result.
        while True:
            signal.pause()


async def _read_exact(
    reader: asyncio.StreamReader,
    size: int,
    deadline: float,
) -> bytes:
    timeout = deadline - time.monotonic()
    if timeout <= 0:
        raise ProviderWatchdogExpired("provider process watchdog expired")
    try:
        return await asyncio.wait_for(
            reader.readexactly(size),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise ProviderWatchdogExpired(
            "provider process watchdog expired"
        ) from exc
    except asyncio.IncompleteReadError as exc:
        raise ProviderProcessError(
            "provider process exited before returning a completion"
        ) from exc


async def _read_frame(
    reader: asyncio.StreamReader,
    deadline: float,
) -> bytes:
    header = await _read_exact(reader, _FRAME_HEADER.size, deadline)
    (size,) = _FRAME_HEADER.unpack(header)
    if size > _MAX_RESULT_BYTES:
        raise ProviderProcessError(
            "provider process returned an oversized completion"
        )
    return await _read_exact(reader, size, deadline)


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        if exc.errno not in (errno.ESRCH, errno.EPERM):
            raise
    try:
        process.kill()
    except ProcessLookupError:
        pass


async def _join_process(process: asyncio.subprocess.Process) -> None:
    """Join despite repeated cancellation of the invoking owner Task."""
    waiter = asyncio.create_task(process.wait())
    interrupted: asyncio.CancelledError | None = None
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError as exc:
            if interrupted is None:
                interrupted = exc
    waiter.result()
    # The original cancellation that entered run_provider_process's finally
    # remains pending and propagates after teardown.  Only a new cancellation
    # received specifically during this join needs to be restored here.
    if interrupted is not None:
        raise interrupted


async def _spawn_provider_process(
    command: list[str],
) -> tuple[asyncio.subprocess.Process, asyncio.CancelledError | None]:
    """Finish an already-started spawn even if its owner is cancelled.

    Cancelling ``create_subprocess_exec`` can otherwise strand a child in the
    small interval after the OS creates it but before asyncio returns its
    Process handle.  Retain the cancellation provenance, obtain the handle,
    then let the caller kill + join before propagating it.
    """
    loop = asyncio.get_running_loop()
    spawning = loop.create_task(
        asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_provider_child_environment(),
            start_new_session=True,
        )
    )
    interrupted: asyncio.CancelledError | None = None
    while not spawning.done():
        try:
            await asyncio.shield(spawning)
        except asyncio.CancelledError as exc:
            if interrupted is None:
                interrupted = exc
    return spawning.result(), interrupted


async def run_provider_process(
    adapter: Any,
    operation: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    watchdog_s: float,
) -> _T:
    """Execute one reviewed adapter operation in a fresh interpreter.

    ``start_new_session`` establishes a private process group before Python
    imports provider code.  On success, failure, timeout, or cancellation the
    parent kills that group and joins the root before decoding or returning
    provider-controlled output.
    """
    if (
        type(watchdog_s) not in (int, float)
        or not math.isfinite(watchdog_s)
        or watchdog_s <= 0
    ):
        raise ValueError("provider process watchdog must be positive")
    request = _build_request(adapter, operation, args, kwargs)

    command = _provider_child_command()
    process, launch_cancellation = await _spawn_provider_process(command)
    with _ACTIVE_PROVIDER_PROCESSES_LOCK:
        _ACTIVE_PROVIDER_PROCESSES.add(process.pid)

    payload: bytes | None = None
    deadline = time.monotonic() + watchdog_s
    try:
        if launch_cancellation is not None:
            raise launch_cancellation
        if process.stdout is None or process.stdin is None:  # pragma: no cover
            raise ProviderProcessError("provider process pipes unavailable")
        ready = await _read_exact(process.stdout, 1, deadline)
        if ready != _READY:
            raise ProviderProcessError(
                "provider process failed its containment handshake"
            )
        process.stdin.write(_FRAME_HEADER.pack(len(request)) + request)
        await process.stdin.drain()
        process.stdin.close()
        payload = await _read_frame(process.stdout, deadline)
    finally:
        _kill_process_group(process)
        try:
            await _join_process(process)
        finally:
            with _ACTIVE_PROVIDER_PROCESSES_LOCK:
                _ACTIVE_PROVIDER_PROCESSES.discard(process.pid)

    if payload is None:  # pragma: no cover - control-flow invariant
        raise ProviderProcessError("provider process returned no completion")
    try:
        return _decode_provider_result(payload)
    except ProviderProcessError:
        raise
    except BaseException as exc:  # noqa: BLE001 - untrusted protocol data
        raise ProviderProcessError(
            "provider process returned an invalid completion"
        ) from exc


if __name__ == "__main__":  # pragma: no cover - exercised through parent IPC
    if sys.argv[1:] != [_CHILD_ARGUMENT]:
        raise SystemExit(2)
    _child_entry()


__all__ = [
    "ProcessProviderRegistration",
    "ProviderChildError",
    "ProviderProcessError",
    "ProviderWatchdogExpired",
    "register_process_provider",
    "run_provider_process",
]
