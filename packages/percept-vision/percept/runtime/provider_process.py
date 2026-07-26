"""One-call exec boundary for effectful vision-provider roots.

The parent owns admission, cognition, and reduction.  A fresh interpreter owns
one provider call and everything it can create: direct ``Task`` objects,
callbacks, executor threads, and subprocess descendants.  The parent does not
decode or publish the result until it has killed the child's process group and
joined the root process.
"""
from __future__ import annotations

import asyncio
import errno
import io
import os
import pickle
import platform
import signal
import stat
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

_T = TypeVar("_T")

_READY = b"R"
_FRAME_HEADER = struct.Struct("!Q")
_MAX_REQUEST_BYTES = 256 * 1024 * 1024
_MAX_RESULT_BYTES = 4 * 1024 * 1024
_CHILD_ARGUMENT = "--percept-provider-child"
_DARWIN_SANDBOX_PROFILE = "(version 1)(allow default)(deny process-fork)"

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


def _provider_child_command() -> list[str]:
    command = [
        sys.executable,
        "-m",
        "percept.runtime.provider_process",
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


def _encode_provider_result(value: Any) -> Any:
    """Convert provider output to an SDK-owned, immutable wire graph."""
    from percept.contracts import Observation, Verdict
    from percept.events import Event

    if type(value) is Verdict:
        from percept.agent import _VerdictSnapshot

        return _VerdictSnapshot.capture(value)
    if type(value) in (list, tuple) and all(
        type(item) is Observation for item in value
    ):
        from percept.agent import _ObservationSnapshot

        return (
            "__percept_observation_snapshots__",
            tuple(_ObservationSnapshot.capture(item) for item in value),
        )
    if type(value) is Event:
        return (
            "__percept_event__",
            str(value.type),
            float(value.t),
            _encode_provider_result(list(value.participants)),
            float(value.confidence),
            None if value.id is None else str(value.id),
        )
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if type(value) is list:
        return (
            "__percept_provider_list__",
            tuple(_encode_provider_result(item) for item in value),
        )
    if type(value) is tuple:
        return tuple(_encode_provider_result(item) for item in value)
    return ("__percept_malformed_provider_result__", type(value).__name__)


class _RestrictedResultUnpickler(pickle.Unpickler):
    """Decode only SDK frozen-result constructors, never provider reducers."""

    _ALLOWED = {
        ("builtins", "frozenset"),
        ("percept.agent", "_FrozenMapping"),
        ("percept.agent", "_ObservationSnapshot"),
        ("percept.agent", "_VerdictSnapshot"),
        ("percept.contracts", "Box"),
        ("percept.contracts", "ClipEvent"),
        ("percept.contracts", "SubjectRef"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(
                f"forbidden provider result constructor {module}.{name}"
            )
        return super().find_class(module, name)


def _decode_provider_value(value: Any) -> Any:
    from percept.agent import _VerdictSnapshot

    if type(value) is _VerdictSnapshot:
        return value.to_verdict()
    if (
        type(value) is tuple
        and len(value) == 2
        and value[0] == "__percept_observation_snapshots__"
        and type(value[1]) is tuple
    ):
        from percept.agent import _ObservationSnapshot

        if not all(type(item) is _ObservationSnapshot for item in value[1]):
            raise ProviderProcessError(
                "provider process returned malformed observation snapshots"
            )
        return [item.to_observation() for item in value[1]]
    if (
        type(value) is tuple
        and len(value) == 2
        and value[0] == "__percept_provider_list__"
        and type(value[1]) is tuple
    ):
        return [_decode_provider_value(item) for item in value[1]]
    if (
        type(value) is tuple
        and len(value) == 6
        and value[0] == "__percept_event__"
    ):
        from percept.events import Event

        participants = _decode_provider_value(value[3])
        if (
            type(value[1]) is not str
            or type(value[2]) not in (int, float)
            or type(participants) is not list
            or type(value[4]) not in (int, float)
            or (value[5] is not None and type(value[5]) is not str)
        ):
            raise ProviderProcessError(
                "provider process returned a malformed event"
            )
        return Event(
            type=value[1],
            t=float(value[2]),
            participants=participants,
            confidence=float(value[4]),
            id=value[5],
        )
    if (
        type(value) is tuple
        and len(value) == 2
        and value[0] == "__percept_malformed_provider_result__"
        and type(value[1]) is str
    ):
        return _MalformedProviderResult(value[1])
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if type(value) is tuple:
        return tuple(_decode_provider_value(item) for item in value)
    raise ProviderProcessError(
        "provider process returned an unsupported completion"
    )


def _decode_provider_result(payload: bytes) -> Any:
    envelope = _RestrictedResultUnpickler(io.BytesIO(payload)).load()
    if (
        type(envelope) is not tuple
        or len(envelope) != 2
        or envelope[0] not in ("ok", "error")
    ):
        raise ProviderProcessError(
            "provider process returned an invalid completion envelope"
        )
    if envelope[0] == "error":
        if type(envelope[1]) is not str:
            raise ProviderProcessError(
                "provider process returned an invalid error envelope"
            )
        raise ProviderChildError(envelope[1])
    return _decode_provider_value(envelope[1])


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


def _child_entry() -> None:
    """Read one pickled call from stdin, execute it, then await parent death."""
    stdout_fd = sys.stdout.fileno()
    try:
        _establish_child_containment()
        _write_all(stdout_fd, _READY)
        payload = _read_frame_fd(sys.stdin.fileno(), limit=_MAX_REQUEST_BYTES)
        callable_obj, args, kwargs = pickle.loads(payload)

        async def invoke() -> Any:
            return await callable_obj(*args, **kwargs)

        try:
            # Stop exactly when the provider root returns.  ``asyncio.run``
            # would perform cooperative task/executor shutdown and can wait
            # forever on resistant descendants.  The OS process boundary,
            # below, is the sole teardown authority.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            value = loop.run_until_complete(invoke())
            envelope: tuple[str, Any] = (
                "ok",
                _encode_provider_result(value),
            )
        except BaseException as exc:  # noqa: BLE001 - safe type-only wire error
            envelope = ("error", type(exc).__name__)
        encoded = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        _send_frame_fd(stdout_fd, encoded, limit=_MAX_RESULT_BYTES)
    except BaseException as exc:  # noqa: BLE001 - best-effort protocol failure
        try:
            encoded = pickle.dumps(
                ("error", type(exc).__name__),
                protocol=pickle.HIGHEST_PROTOCOL,
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
    callable_obj: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    watchdog_s: float,
) -> _T:
    """Execute one picklable async call in a fresh interpreter.

    ``start_new_session`` establishes a private process group before Python
    imports provider code.  On success, failure, timeout, or cancellation the
    parent kills that group and joins the root before decoding or returning
    provider-controlled output.
    """
    if watchdog_s <= 0:
        raise ValueError("provider process watchdog must be positive")
    try:
        request = pickle.dumps(
            (callable_obj, args, kwargs),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    except BaseException as exc:  # noqa: BLE001 - fail closed at the boundary
        raise ProviderProcessError(
            "provider call is not serializable for process isolation"
        ) from exc
    if len(request) > _MAX_REQUEST_BYTES:
        raise ProviderProcessError(
            "provider request exceeded the process boundary limit"
        )

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
    "ProviderChildError",
    "ProviderProcessError",
    "ProviderWatchdogExpired",
    "run_provider_process",
]
