"""Reference controller for protected Hosted BYOK Linux/amd64 qualification.

This file is not a self-authenticating root launcher.  A separately reviewed
release authority must:

* install an exact reviewed qualification commit in the protected bare repo;
* materialize this exact controller at ``CONTROLLER_PATH``;
* install the root-only policy at ``POLICY_PATH``;
* invoke ``/usr/bin/python3 -I -S CONTROLLER_PATH`` with ``CLEAN_ENV``; and
* destroy the ephemeral Docker authority after every outcome, including
  SIGKILL or host loss.

The fixed protected policy is the external trust root.  There are no command
line, source, result, tool, daemon, temp, or evidence-path inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence


SCHEMA = 3
POLICY_SCHEMA = 1
PLATFORM = "linux/amd64"
BASE_DIGEST = (
    "sha256:"
    "cab2dbf575e971934a81e4622f5aba17aa7929719bd7e31033a3a83b97fd0464"
)
ACCEPTED_PRODUCTION_SHA = "39d51380f6a3c1e923f0f72dd94dcee3a176693c"
ACCEPTED_PRODUCTION_TREE = "a69e8c0f727e7ca8986722fb1d5218b9cb7d2a60"
CONTROLLER_PATH = Path(
    "/opt/percept-native-authority/v1/qualify_hosted_byok_native.py"
)
RECIPE_PATH = Path("/opt/percept-native-authority/v1/Dockerfile.hosted-byok-native")
RUNNER_PATH = Path(
    "/opt/percept-native-authority/v1/hosted_byok_native_matrix.py"
)
POLICY_PATH = Path("/etc/percept/native-authority-v1.json")
SOURCE_REPOSITORY = Path("/var/lib/percept-native-authority/source.git")
WORK_ROOT = Path("/var/lib/percept-native-authority/work")
PENDING_EVIDENCE_ROOT = Path("/var/lib/percept-native-authority/pending")
FIXED_GIT = Path("/usr/bin/git")
FIXED_PYTHON = Path("/usr/bin/python3")
FIXED_DOCKER = Path("/usr/bin/docker")
FIXED_DOCKER_SOCKET = Path("/run/docker.sock")
CLEAN_ENV = {
    "PATH": "/usr/bin:/bin",
    "LC_ALL": "C",
    "LANG": "C",
    "HOME": "/var/empty",
    "TMPDIR": str(WORK_ROOT),
}
CASES = tuple(
    f"{provider}:{scenario}"
    for provider in ("gemini", "openrouter")
    for scenario in ("success", "deadline", "cancel")
)
ALLOWED_QUALIFICATION_PATHS = frozenset(
    {
        "packages/percept-vision/Dockerfile.hosted-byok-native",
        "packages/percept-vision/HOSTED_BYOK_NATIVE_QUALIFICATION.md",
        "packages/percept-vision/tests/test_hosted_byok_native_authority.py",
        "packages/percept-vision/tools/hosted_byok_native_matrix.py",
        "packages/percept-vision/tools/qualify_hosted_byok_native.py",
    }
)
CONTEXT_PATHS = (
    "packages/percept-vision/Dockerfile.hosted-byok-native",
    "packages/percept-vision/percept",
    "packages/percept-vision/tests/test_hosted_byok.py",
    "packages/percept-vision/tools/hosted_byok_native_matrix.py",
)
LABEL_PREFIX = "org.percept.hosted-byok."
MAX_CONTEXT_FILES = 512
MAX_CONTEXT_FILE_BYTES = 16 * 1024 * 1024
MAX_CONTEXT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_DOCKERFILE = """# Linux/amd64 Hosted BYOK qualification image only.
#
# A separately protected release authority authenticates this exact recipe and
# the Git blobs copied below before invoking Docker. There are no installers,
# mutable tags, build-time network consumers, or shell entrypoints.
FROM --platform=linux/amd64 python:3.12-slim@sha256:cab2dbf575e971934a81e4622f5aba17aa7929719bd7e31033a3a83b97fd0464

ARG SOURCE_SHA
ARG SOURCE_TREE
ARG PRODUCTION_SHA
ARG PRODUCTION_TREE
ARG CONTEXT_SHA256
ARG RECIPE_SHA256
LABEL org.percept.hosted-byok.source-sha="${SOURCE_SHA}" \\
      org.percept.hosted-byok.source-tree="${SOURCE_TREE}" \\
      org.percept.hosted-byok.production-sha="${PRODUCTION_SHA}" \\
      org.percept.hosted-byok.production-tree="${PRODUCTION_TREE}" \\
      org.percept.hosted-byok.context-sha256="${CONTEXT_SHA256}" \\
      org.percept.hosted-byok.recipe-sha256="${RECIPE_SHA256}" \\
      org.percept.hosted-byok.base-digest="sha256:cab2dbf575e971934a81e4622f5aba17aa7929719bd7e31033a3a83b97fd0464" \\
      org.percept.hosted-byok.platform="linux/amd64"

WORKDIR /qualification
COPY percept /qualification/percept
COPY tests/test_hosted_byok.py /qualification/tests/test_hosted_byok.py
COPY tools/hosted_byok_native_matrix.py /qualification/hosted_byok_native_matrix.py

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUTF8=1

ENTRYPOINT ["/usr/local/bin/python", "-I", "/qualification/hosted_byok_native_matrix.py"]
"""


class QualificationRejected(RuntimeError):
    """Authenticated input or observed execution violated the contract."""


class QualificationUnavailable(RuntimeError):
    """The protected authority is absent or cannot prove completion."""


class AuthorityInterrupted(BaseException):
    """A catchable process signal interrupted qualification."""


_authority_signal_latch = threading.Lock()
_first_authority_interruption: AuthorityInterrupted | None = None
_owned_command_active = False


def _raise_first_authority_interruption() -> None:
    if _first_authority_interruption is not None:
        raise _first_authority_interruption


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    timeout_s: float
    max_output: int = MAX_OUTPUT_BYTES


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class FileIdentity:
    path: str
    sha256: str
    uid: int
    gid: int
    mode: int
    device: int
    inode: int


@dataclass(frozen=True)
class SourceIdentity:
    commit: str
    tree: str


@dataclass(frozen=True)
class ContextIdentity:
    sha256: str
    recipe_sha256: str
    runner_sha256: str
    file_count: int
    byte_count: int


@dataclass(frozen=True)
class AuthorityPolicy:
    qualification_commit: str
    qualification_tree: str
    controller_sha256: str
    recipe_sha256: str
    runner_sha256: str
    git: FileIdentity
    python: FileIdentity
    docker: FileIdentity
    socket: FileIdentity
    daemon_id: str
    daemon_version: str
    docker_root_dir: str
    runner_id: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json(data: bytes, *, cap: int = MAX_OUTPUT_BYTES) -> Any:
    if len(data) > cap:
        raise QualificationRejected("JSON exceeds the bounded size")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise QualificationRejected(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def invalid_constant(value: str) -> None:
        raise QualificationRejected(f"invalid JSON constant: {value}")

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationRejected("value is not one strict JSON document") from exc


def _exact_object(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise QualificationRejected(f"{name} has an unexpected schema")
    return value


def _exact_string(value: Any, pattern: re.Pattern[str] | None, name: str) -> str:
    if type(value) is not str or not value or (
        pattern is not None and not pattern.fullmatch(value)
    ):
        raise QualificationRejected(f"{name} is invalid")
    return value


def _parse_file_identity(
    value: Any,
    expected_path: Path,
    name: str,
    *,
    socket: bool = False,
) -> FileIdentity:
    item = _exact_object(
        value,
        {"path", "sha256", "uid", "gid", "mode", "device", "inode"},
        name,
    )
    if item["path"] != str(expected_path):
        raise QualificationRejected(f"{name} path is not fixed")
    for field in ("uid", "gid", "mode", "device", "inode"):
        if type(item[field]) is not int or item[field] < 0:
            raise QualificationRejected(f"{name} {field} is invalid")
    return FileIdentity(
        path=item["path"],
        sha256=(
            ""
            if socket and item["sha256"] == ""
            else _exact_string(item["sha256"], HEX64, f"{name} sha256")
        ),
        uid=item["uid"],
        gid=item["gid"],
        mode=item["mode"],
        device=item["device"],
        inode=item["inode"],
    )


def parse_policy(value: Any) -> AuthorityPolicy:
    root = _exact_object(
        value,
        {
            "schema",
            "qualification",
            "accepted_production",
            "tools",
            "daemon",
            "authority",
        },
        "authority policy",
    )
    if type(root["schema"]) is not int or root["schema"] != POLICY_SCHEMA:
        raise QualificationRejected("authority policy schema is invalid")
    qualification = _exact_object(
        root["qualification"],
        {
            "commit",
            "tree",
            "controller_sha256",
            "recipe_sha256",
            "runner_sha256",
        },
        "qualification policy",
    )
    accepted = _exact_object(
        root["accepted_production"], {"commit", "tree"}, "production policy"
    )
    if accepted != {
        "commit": ACCEPTED_PRODUCTION_SHA,
        "tree": ACCEPTED_PRODUCTION_TREE,
    }:
        raise QualificationRejected("accepted production policy is not exact")
    tools = _exact_object(
        root["tools"], {"git", "python", "docker", "socket"}, "tool policy"
    )
    daemon = _exact_object(
        root["daemon"],
        {"id", "version", "docker_root_dir"},
        "daemon policy",
    )
    authority = _exact_object(
        root["authority"],
        {"runner_id", "ephemeral_daemon", "destroy_on_every_outcome"},
        "authority policy",
    )
    if (
        authority["ephemeral_daemon"] is not True
        or authority["destroy_on_every_outcome"] is not True
    ):
        raise QualificationRejected("authority must own an ephemeral daemon")
    return AuthorityPolicy(
        qualification_commit=_exact_string(
            qualification["commit"], HEX40, "qualification commit"
        ),
        qualification_tree=_exact_string(
            qualification["tree"], HEX40, "qualification tree"
        ),
        controller_sha256=_exact_string(
            qualification["controller_sha256"], HEX64, "controller digest"
        ),
        recipe_sha256=_exact_string(
            qualification["recipe_sha256"], HEX64, "recipe digest"
        ),
        runner_sha256=_exact_string(
            qualification["runner_sha256"], HEX64, "runner digest"
        ),
        git=_parse_file_identity(tools["git"], FIXED_GIT, "git"),
        python=_parse_file_identity(tools["python"], FIXED_PYTHON, "python"),
        docker=_parse_file_identity(tools["docker"], FIXED_DOCKER, "docker"),
        socket=_parse_file_identity(
            tools["socket"],
            FIXED_DOCKER_SOCKET,
            "docker socket",
            socket=True,
        ),
        daemon_id=_exact_string(daemon["id"], None, "daemon id"),
        daemon_version=_exact_string(daemon["version"], None, "daemon version"),
        docker_root_dir=_exact_string(
            daemon["docker_root_dir"], None, "docker root dir"
        ),
        runner_id=_exact_string(authority["runner_id"], None, "authority runner id"),
    )


def _path_components(path: Path) -> tuple[Path, ...]:
    absolute = path if path.is_absolute() else path.resolve()
    parts = absolute.parts
    return tuple(Path(*parts[:index]) for index in range(1, len(parts) + 1))


def protected_identity(path: Path, *, kind: str) -> FileIdentity:
    for component in _path_components(path):
        try:
            info = component.lstat()
        except OSError as exc:
            raise QualificationUnavailable(f"protected path unavailable: {component}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise QualificationUnavailable(f"protected path is a symlink: {component}")
        if component != path:
            if not stat.S_ISDIR(info.st_mode):
                raise QualificationUnavailable(
                    f"protected ancestor is not a directory: {component}"
                )
            if info.st_uid != 0 or info.st_mode & (
                stat.S_IWGRP | stat.S_IWOTH
            ):
                raise QualificationUnavailable(
                    f"protected ancestor is mutable: {component}"
                )
    info = path.lstat()
    expected = {
        "socket": stat.S_ISSOCK,
        "file": stat.S_ISREG,
        "directory": stat.S_ISDIR,
    }.get(kind)
    if expected is None:
        raise ValueError(f"unsupported protected path kind: {kind}")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise QualificationUnavailable(f"protected {kind} is group/world writable: {path}")
    if not expected(info.st_mode) or info.st_uid != 0:
        raise QualificationUnavailable(f"protected {kind} is not root-owned: {path}")
    digest = "" if kind in {"socket", "directory"} else _sha256(path.read_bytes())
    return FileIdentity(
        path=str(path),
        sha256=digest,
        uid=info.st_uid,
        gid=info.st_gid,
        mode=stat.S_IMODE(info.st_mode),
        device=info.st_dev,
        inode=info.st_ino,
    )


def _same_identity(actual: FileIdentity, expected: FileIdentity, name: str) -> None:
    if actual != expected:
        raise QualificationUnavailable(f"protected {name} identity changed")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    def group_exists() -> bool:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def joined() -> bool:
        return process.poll() is not None and not group_exists()

    def wait_joined(timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while not joined():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        process.wait()
        return True

    if joined():
        process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if wait_joined(1.0):
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if not wait_joined(1.0):
        raise QualificationUnavailable(
            "owned protected command process group could not be joined"
        )


def _read_streams_bounded(
    process: subprocess.Popen[bytes], command: Command
) -> Completed:
    selector: selectors.BaseSelector | None = None
    streams = (process.stdout, process.stderr)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + command.timeout_s
    failed = False
    try:
        _raise_first_authority_interruption()
        if process.stdout is None or process.stderr is None:
            raise QualificationUnavailable("protected command pipes are unavailable")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            _raise_first_authority_interruption()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QualificationUnavailable("protected command timed out")
            events = selector.select(min(remaining, 0.25))
            _raise_first_authority_interruption()
            if not events and process.poll() is not None:
                events = [
                    (key, selectors.EVENT_READ)
                    for key in tuple(selector.get_map().values())
                ]
            for key, _ in events:
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = buffers[key.data]
                target.extend(chunk)
                if len(target) > command.max_output:
                    raise QualificationRejected("protected command output exceeded cap")
        returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        _raise_first_authority_interruption()
    except BaseException:
        failed = True
        quiescence_error: BaseException | None = None
        try:
            _terminate_process_group(process)
        except BaseException as exc:
            quiescence_error = exc
        if quiescence_error is not None:
            raise QualificationUnavailable(
                "owned protected command quiescence could not be proven"
            ) from quiescence_error
        _raise_first_authority_interruption()
        raise
    finally:
        close_error: BaseException | None = None
        if selector is not None:
            try:
                selector.close()
            except BaseException as exc:
                close_error = exc
        for stream in streams:
            if stream is None:
                continue
            try:
                stream.close()
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
        if close_error is not None and not failed:
            raise close_error
    return Completed(returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"]))


def run_command(
    command: Command,
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> Completed:
    global _owned_command_active

    if _owned_command_active:
        raise QualificationUnavailable("protected command boundary is already active")
    _owned_command_active = True
    process: subprocess.Popen[bytes] | None = None
    completed: Completed | None = None
    command_error: BaseException | None = None
    quiescence_error: BaseException | None = None
    try:
        _raise_first_authority_interruption()
        process = subprocess.Popen(
            command.argv,
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
        completed = _read_streams_bounded(process, command)
    except BaseException as exc:
        command_error = exc
    finally:
        if process is not None:
            try:
                _terminate_process_group(process)
            except BaseException as exc:
                quiescence_error = exc
        _owned_command_active = False

    if quiescence_error is not None:
        raise QualificationUnavailable(
            "owned protected command quiescence could not be proven"
        ) from quiescence_error
    _raise_first_authority_interruption()
    if command_error is not None:
        raise command_error
    if completed is None:
        raise QualificationUnavailable("protected command produced no result")
    return completed


def _git(
    args: Sequence[str],
    *,
    policy: AuthorityPolicy,
    max_output: int = 2**20,
) -> Completed:
    _same_identity(protected_identity(FIXED_GIT, kind="file"), policy.git, "git")
    completed = run_command(
        Command(
            (
                str(FIXED_GIT),
                "-c",
                "core.fsmonitor=false",
                f"--git-dir={SOURCE_REPOSITORY}",
                *args,
            ),
            timeout_s=30,
            max_output=max_output,
        ),
        cwd=WORK_ROOT,
        env=CLEAN_ENV,
    )
    _same_identity(protected_identity(FIXED_GIT, kind="file"), policy.git, "git")
    return completed


def _git_text(args: Sequence[str], *, policy: AuthorityPolicy) -> str:
    completed = _git(args, policy=policy)
    if completed.returncode != 0:
        raise QualificationRejected("protected Git command failed")
    try:
        return completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise QualificationRejected("protected Git output is not ASCII") from exc


def authenticate_source(policy: AuthorityPolicy) -> SourceIdentity:
    commit = _git_text(
        ("rev-parse", "--verify", f"{policy.qualification_commit}^{{commit}}"),
        policy=policy,
    )
    tree = _git_text(
        ("rev-parse", "--verify", f"{policy.qualification_commit}^{{tree}}"),
        policy=policy,
    )
    parent = _git_text(
        ("rev-parse", "--verify", f"{policy.qualification_commit}^"),
        policy=policy,
    )
    production_tree = _git_text(
        ("rev-parse", "--verify", f"{ACCEPTED_PRODUCTION_SHA}^{{tree}}"),
        policy=policy,
    )
    if (
        commit != policy.qualification_commit
        or tree != policy.qualification_tree
        or parent != ACCEPTED_PRODUCTION_SHA
        or production_tree != ACCEPTED_PRODUCTION_TREE
    ):
        raise QualificationRejected("reviewed qualification Git object is not exact")
    changed = _git(
        (
            "diff",
            "--name-only",
            "-z",
            ACCEPTED_PRODUCTION_SHA,
            commit,
            "--",
        ),
        policy=policy,
    )
    if changed.returncode != 0:
        raise QualificationRejected("unable to compare qualification source")
    try:
        paths = {
            item.decode("utf-8")
            for item in changed.stdout.split(b"\0")
            if item
        }
    except UnicodeDecodeError as exc:
        raise QualificationRejected("qualification path is not UTF-8") from exc
    if paths != ALLOWED_QUALIFICATION_PATHS:
        raise QualificationRejected("qualification path set is not exact")
    return SourceIdentity(commit=commit, tree=tree)


def _tree_records(data: bytes) -> list[tuple[str, str, str]]:
    records: list[tuple[str, str, str]] = []
    for raw in data.split(b"\0"):
        if not raw:
            continue
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, kind, oid = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise QualificationRejected("malformed Git tree record") from exc
        if mode not in {"100644", "100755"} or kind != "blob":
            raise QualificationRejected(f"unsupported context tree entry: {path}")
        if (
            path.startswith("/")
            or ".." in Path(path).parts
            or not path.startswith("packages/percept-vision/")
        ):
            raise QualificationRejected("unsafe context path")
        records.append((mode, oid, path))
    paths = [path for _, _, path in records]
    if (
        not records
        or len(records) > MAX_CONTEXT_FILES
        or len(paths) != len(set(paths))
    ):
        raise QualificationRejected("context file set is invalid")
    return records


def construct_context(
    source: SourceIdentity,
    destination: Path,
    *,
    policy: AuthorityPolicy,
) -> ContextIdentity:
    tree = _git(
        ("ls-tree", "-r", "-z", "--full-tree", source.commit, "--", *CONTEXT_PATHS),
        policy=policy,
        max_output=2**20,
    )
    if tree.returncode != 0:
        raise QualificationRejected("unable to enumerate exact build context")
    records = _tree_records(tree.stdout)
    total = 0
    manifest: list[dict[str, Any]] = []
    prefix = "packages/percept-vision/"
    for mode, oid, path in records:
        size_text = _git_text(("cat-file", "-s", oid), policy=policy)
        try:
            size = int(size_text)
        except ValueError as exc:
            raise QualificationRejected("Git blob size is invalid") from exc
        if size < 0 or size > MAX_CONTEXT_FILE_BYTES:
            raise QualificationRejected(f"context blob exceeds cap: {path}")
        total += size
        if total > MAX_CONTEXT_BYTES:
            raise QualificationRejected("context exceeds total cap")
        blob = _git(
            ("cat-file", "blob", oid),
            policy=policy,
            max_output=max(1, size),
        )
        if blob.returncode != 0 or len(blob.stdout) != size:
            raise QualificationRejected(f"unable to read exact Git blob: {path}")
        relative = path.removeprefix(prefix)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_bytes(blob.stdout)
        target.chmod(0o755 if mode == "100755" else 0o644)
        manifest.append(
            {
                "path": relative,
                "mode": mode,
                "oid": oid,
                "size": size,
                "sha256": _sha256(blob.stdout),
            }
        )
    materialized = {item["path"] for item in manifest}
    required = {
        "Dockerfile.hosted-byok-native",
        "tests/test_hosted_byok.py",
        "tools/hosted_byok_native_matrix.py",
    }
    if not required.issubset(materialized) or not any(
        path.startswith("percept/") for path in materialized
    ):
        raise QualificationRejected("qualification context is incomplete")
    recipe = (destination / "Dockerfile.hosted-byok-native").read_bytes()
    runner = (
        destination / "tools" / "hosted_byok_native_matrix.py"
    ).read_bytes()
    if recipe != EXPECTED_DOCKERFILE.encode("utf-8"):
        raise QualificationRejected("native recipe differs from reviewed controller")
    if (
        _sha256(recipe) != policy.recipe_sha256
        or _sha256(runner) != policy.runner_sha256
    ):
        raise QualificationRejected("reviewed recipe/runner digest mismatch")
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return ContextIdentity(
        sha256=_sha256(manifest_bytes),
        recipe_sha256=_sha256(recipe),
        runner_sha256=_sha256(runner),
        file_count=len(manifest),
        byte_count=total,
    )


def expected_labels(
    source: SourceIdentity, context: ContextIdentity
) -> dict[str, str]:
    return {
        f"{LABEL_PREFIX}source-sha": source.commit,
        f"{LABEL_PREFIX}source-tree": source.tree,
        f"{LABEL_PREFIX}production-sha": ACCEPTED_PRODUCTION_SHA,
        f"{LABEL_PREFIX}production-tree": ACCEPTED_PRODUCTION_TREE,
        f"{LABEL_PREFIX}context-sha256": context.sha256,
        f"{LABEL_PREFIX}recipe-sha256": context.recipe_sha256,
        f"{LABEL_PREFIX}base-digest": BASE_DIGEST,
        f"{LABEL_PREFIX}platform": PLATFORM,
    }


def validate_image(
    value: Any, *, image_id: str, labels: Mapping[str, str]
) -> None:
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        raise QualificationRejected("image inspect must contain one image")
    image = value[0]
    if (
        image.get("Id") != image_id
        or image.get("Os") != "linux"
        or image.get("Architecture") != "amd64"
    ):
        raise QualificationRejected("image identity/platform mismatch")
    config = image.get("Config")
    if type(config) is not dict:
        raise QualificationRejected("image config is missing")
    actual_labels = config.get("Labels")
    if type(actual_labels) is not dict:
        raise QualificationRejected("image labels are missing")
    scoped = {
        key: item
        for key, item in actual_labels.items()
        if type(key) is str and key.startswith(LABEL_PREFIX)
    }
    if scoped != dict(labels):
        raise QualificationRejected("image label set is not exact")
    if config.get("Entrypoint") != [
        "/usr/local/bin/python",
        "-I",
        "/qualification/hosted_byok_native_matrix.py",
    ]:
        raise QualificationRejected("image entrypoint is not exact")
    if (
        config.get("Cmd") not in (None, [])
        or config.get("WorkingDir") != "/qualification"
    ):
        raise QualificationRejected("image command/workdir is not exact")


def validate_container(
    value: Any,
    *,
    container_id: str,
    image_id: str,
    name: str,
) -> None:
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        raise QualificationRejected("container inspect must contain one container")
    container = value[0]
    if (
        container.get("Id") != container_id
        or container.get("Image") != image_id
        or container.get("Name") != f"/{name}"
    ):
        raise QualificationRejected("container identity is not exact")
    host = container.get("HostConfig")
    config = container.get("Config")
    if type(host) is not dict or type(config) is not dict:
        raise QualificationRejected("container config is missing")
    expected = {
        "NetworkMode": "none",
        "ReadonlyRootfs": True,
        "PidsLimit": 128,
        "Memory": 768 * 1024 * 1024,
        "NanoCpus": 2_000_000_000,
    }
    if any(host.get(key) != item for key, item in expected.items()):
        raise QualificationRejected("container resource boundary is not exact")
    if (
        host.get("CapDrop") != ["ALL"]
        or host.get("SecurityOpt") != ["no-new-privileges"]
        or config.get("User") != "65534:65534"
    ):
        raise QualificationRejected("container privilege boundary is not exact")
    tmpfs = host.get("Tmpfs")
    if type(tmpfs) is not dict or tmpfs.get("/tmp") != (
        "rw,noexec,nosuid,nodev,size=134217728"
    ):
        raise QualificationRejected("container tmpfs boundary is not exact")


def validate_native_result(value: Any) -> dict[str, Any]:
    root = _exact_object(
        value,
        {
            "schema",
            "status",
            "platform",
            "architecture",
            "python",
            "cases",
            "observations",
            "environment_poisoned",
            "no_descendants_observed",
            "joined_observed",
            "late_effect_observed",
        },
        "native result",
    )
    cases = root["cases"]
    if (
        type(root["schema"]) is not int
        or root["schema"] != SCHEMA
        or root["status"] != "passed"
        or root["platform"] != "linux"
        or root["architecture"] != "amd64"
        or root["python"] != "3.12"
        or type(cases) is not list
        or any(type(case) is not str for case in cases)
        or cases != list(CASES)
        or len(set(cases)) != len(CASES)
    ):
        raise QualificationRejected("native result header is not exact")
    observations = root["observations"]
    if type(observations) is not list or len(observations) != len(CASES):
        raise QualificationRejected("native result observations are incomplete")
    for expected_case, observation in zip(CASES, observations):
        item = _exact_object(
            observation,
            {
                "provider",
                "scenario",
                "outcome",
                "provider_called",
                "process_exit_observed",
                "pids_after_return",
                "pids_after_horizon",
                "observed_horizon_ms",
                "late_effect_paths",
            },
            "native observation",
        )
        provider, scenario = expected_case.split(":", 1)
        if (
            item["provider"] != provider
            or item["scenario"] != scenario
            or item["outcome"]
            != {
                "success": "success",
                "deadline": "deadline",
                "cancel": "cancelled",
            }[scenario]
            or item["provider_called"] is not True
            or item["process_exit_observed"] is not True
            or item["pids_after_return"] != []
            or item["pids_after_horizon"] != []
            or type(item["observed_horizon_ms"]) is not int
            or item["observed_horizon_ms"] < 2500
            or item["late_effect_paths"] != []
        ):
            raise QualificationRejected("native observation is not exact")
    if (
        root["environment_poisoned"] is not True
        or root["no_descendants_observed"] is not True
        or root["joined_observed"] is not True
        or root["late_effect_observed"] is not False
    ):
        raise QualificationRejected("native containment observation failed")
    return root


class ProtectedDockerAuthority:
    """Docker client bound to a root policy and ephemeral daemon identity."""

    def __init__(self, policy: AuthorityPolicy, config_dir: Path) -> None:
        self.policy = policy
        self.config_dir = config_dir
        config_dir.mkdir(mode=0o700)
        self.env = {
            **CLEAN_ENV,
            "HOME": str(config_dir),
            "DOCKER_CONFIG": str(config_dir),
            "DOCKER_HOST": f"unix://{FIXED_DOCKER_SOCKET}",
            "DOCKER_BUILDKIT": "0",
            "DOCKER_CLI_HINTS": "false",
        }
        self._check()

    def _check(self) -> None:
        _same_identity(
            protected_identity(FIXED_DOCKER, kind="file"),
            self.policy.docker,
            "docker",
        )
        _same_identity(
            protected_identity(FIXED_DOCKER_SOCKET, kind="socket"),
            self.policy.socket,
            "docker socket",
        )

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        timeout_s: float,
        max_output: int = MAX_OUTPUT_BYTES,
    ) -> Completed:
        self._check()
        result = run_command(
            Command(
                (
                    str(FIXED_DOCKER),
                    "--host",
                    f"unix://{FIXED_DOCKER_SOCKET}",
                    "--config",
                    str(self.config_dir),
                    *args,
                ),
                timeout_s=timeout_s,
                max_output=max_output,
            ),
            cwd=cwd,
            env=self.env,
        )
        self._check()
        return result

    def authenticate_daemon(self) -> dict[str, Any]:
        version = self.run(
            ("version", "--format", "{{json .Server}}"),
            cwd=self.config_dir,
            timeout_s=15,
        )
        info_result = self.run(
            ("info", "--format", "{{json .}}"),
            cwd=self.config_dir,
            timeout_s=15,
        )
        if version.returncode != 0 or info_result.returncode != 0:
            raise QualificationUnavailable("ephemeral Docker daemon is unavailable")
        server = _strict_json(version.stdout.strip())
        info = _strict_json(info_result.stdout.strip())
        if (
            type(server) is not dict
            or type(info) is not dict
            or server.get("Os") != "linux"
            or server.get("Arch") not in {"amd64", "x86_64"}
            or info.get("OSType") != "linux"
            or info.get("Architecture") not in {"amd64", "x86_64"}
            or info.get("ID") != self.policy.daemon_id
            or info.get("ServerVersion") != self.policy.daemon_version
            or info.get("DockerRootDir") != self.policy.docker_root_dir
        ):
            raise QualificationUnavailable("ephemeral Docker daemon identity is not exact")
        return {"server": server, "info": info}

    def cleanup_container(self, reference: str, cwd: Path) -> None:
        self.run(("container", "rm", "--force", reference), cwd=cwd, timeout_s=30)
        check = self.run(
            ("container", "inspect", reference), cwd=cwd, timeout_s=15
        )
        if check.returncode == 0:
            raise QualificationRejected("qualification container cleanup was not joined")

    def cleanup_image(self, reference: str, cwd: Path) -> None:
        self.run(("image", "rm", "--force", reference), cwd=cwd, timeout_s=30)
        check = self.run(("image", "inspect", reference), cwd=cwd, timeout_s=15)
        if check.returncode == 0:
            raise QualificationRejected("qualification image cleanup was not joined")


def build_and_run(
    *,
    authority: ProtectedDockerAuthority,
    context_dir: Path,
    source: SourceIdentity,
    context: ContextIdentity,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    labels = expected_labels(source, context)
    token = os.urandom(8).hex()
    tag = f"percept-hosted-byok-native:{source.commit[:12]}-{token}"
    container_name = f"percept-hosted-byok-native-{source.commit[:12]}-{token}"
    iidfile = context_dir.parent / "image-id"
    image_id: str | None = None
    container_id: str | None = None
    run_succeeded = False
    cleanup_errors: list[BaseException] = []
    try:
        build_args: list[str] = []
        for key, value in (
            ("SOURCE_SHA", source.commit),
            ("SOURCE_TREE", source.tree),
            ("PRODUCTION_SHA", ACCEPTED_PRODUCTION_SHA),
            ("PRODUCTION_TREE", ACCEPTED_PRODUCTION_TREE),
            ("CONTEXT_SHA256", context.sha256),
            ("RECIPE_SHA256", context.recipe_sha256),
        ):
            build_args.extend(("--build-arg", f"{key}={value}"))
        build = authority.run(
            (
                "build",
                "--platform",
                PLATFORM,
                "--network",
                "none",
                "--no-cache",
                "--pull=false",
                "--file",
                "Dockerfile.hosted-byok-native",
                "--iidfile",
                str(iidfile),
                "--tag",
                tag,
                *build_args,
                ".",
            ),
            cwd=context_dir,
            timeout_s=300,
        )
        if build.returncode != 0:
            raise QualificationUnavailable("protected native build failed")
        try:
            image_id = iidfile.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise QualificationRejected("build did not emit an image id") from exc
        if not IMAGE_ID.fullmatch(image_id):
            raise QualificationRejected("build emitted a malformed image id")
        image_inspect = authority.run(
            ("image", "inspect", image_id), cwd=context_dir, timeout_s=30
        )
        if image_inspect.returncode != 0:
            raise QualificationRejected("image inspect failed")
        image_value = _strict_json(image_inspect.stdout)
        validate_image(image_value, image_id=image_id, labels=labels)

        create = authority.run(
            (
                "container",
                "create",
                "--name",
                container_name,
                "--platform",
                PLATFORM,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "128",
                "--memory",
                "768m",
                "--cpus",
                "2",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=128m",
                "--user",
                "65534:65534",
                image_id,
            ),
            cwd=context_dir,
            timeout_s=30,
        )
        if create.returncode != 0:
            raise QualificationRejected("qualification container create failed")
        container_id = create.stdout.decode("ascii").strip()
        if not CONTAINER_ID.fullmatch(container_id):
            raise QualificationRejected("container create emitted a malformed id")
        container_inspect = authority.run(
            ("container", "inspect", container_id),
            cwd=context_dir,
            timeout_s=30,
        )
        if container_inspect.returncode != 0:
            raise QualificationRejected("container inspect failed")
        validate_container(
            _strict_json(container_inspect.stdout),
            container_id=container_id,
            image_id=image_id,
            name=container_name,
        )
        native = authority.run(
            ("container", "start", "--attach", container_id),
            cwd=context_dir,
            timeout_s=180,
        )
        if native.returncode != 0 or native.stderr:
            raise QualificationRejected("native matrix process failed or wrote stderr")
        lines = native.stdout.splitlines()
        if len(lines) != 1:
            raise QualificationRejected("native matrix stdout is not one frame")
        result = validate_native_result(_strict_json(lines[0]))
        state = authority.run(
            ("container", "inspect", container_id, "--format", "{{json .State}}"),
            cwd=context_dir,
            timeout_s=15,
        )
        state_value = _strict_json(state.stdout.strip())
        if (
            state.returncode != 0
            or type(state_value) is not dict
            or state_value.get("Running") is not False
            or state_value.get("ExitCode") != 0
            or type(state_value.get("FinishedAt")) is not str
            or not state_value["FinishedAt"]
        ):
            raise QualificationRejected("container terminal state is not authenticated")
        run_succeeded = True
        return image_id, result, {
            "labels": labels,
            "image_inspect": image_value[0],
            "container_id": container_id,
            "terminal_state": state_value,
        }
    finally:
        if container_id is not None:
            try:
                authority.cleanup_container(container_id, context_dir)
            except BaseException as exc:
                cleanup_errors.append(exc)
        else:
            try:
                authority.cleanup_container(container_name, context_dir)
            except BaseException:
                # A create that never committed is expected to be absent.
                pass
        if image_id is not None:
            try:
                authority.cleanup_image(image_id, context_dir)
            except BaseException as exc:
                cleanup_errors.append(exc)
        else:
            try:
                authority.cleanup_image(tag, context_dir)
            except BaseException:
                # The external ephemeral daemon is the final crash boundary.
                pass
        if cleanup_errors:
            raise QualificationRejected("qualification cleanup could not be proven") from (
                cleanup_errors[0]
            )
        if not run_succeeded and image_id is not None:
            # The function must never convert a failed execution into evidence.
            pass


def _load_policy() -> AuthorityPolicy:
    identity = protected_identity(POLICY_PATH, kind="file")
    if identity.mode & 0o077:
        raise QualificationUnavailable("authority policy is not root-only")
    return parse_policy(_strict_json(POLICY_PATH.read_bytes()))


def authenticate_bootstrap(policy: AuthorityPolicy) -> None:
    if (
        os.geteuid() != 0
        or Path(sys.executable) != FIXED_PYTHON
        or not sys.flags.isolated
        or not sys.flags.no_site
        or tuple(sys.argv) != (str(CONTROLLER_PATH),)
        or dict(os.environ) != CLEAN_ENV
        or Path(__file__) != CONTROLLER_PATH
    ):
        raise QualificationUnavailable("protected isolated bootstrap is not exact")
    _same_identity(
        protected_identity(FIXED_PYTHON, kind="file"), policy.python, "python"
    )
    _same_identity(protected_identity(FIXED_GIT, kind="file"), policy.git, "git")
    controller = protected_identity(CONTROLLER_PATH, kind="file")
    recipe = protected_identity(RECIPE_PATH, kind="file")
    runner = protected_identity(RUNNER_PATH, kind="file")
    if (
        controller.sha256 != policy.controller_sha256
        or recipe.sha256 != policy.recipe_sha256
        or runner.sha256 != policy.runner_sha256
    ):
        raise QualificationUnavailable("materialized qualification bundle is not exact")
    protected_identity(SOURCE_REPOSITORY, kind="directory")
    protected_identity(WORK_ROOT, kind="directory")
    protected_identity(PENDING_EVIDENCE_ROOT, kind="directory")


def _safe_write_evidence(commit: str, evidence: Mapping[str, Any]) -> Path:
    protected_identity(PENDING_EVIDENCE_ROOT, kind="directory")
    path = PENDING_EVIDENCE_ROOT / f"{commit}.json"
    data = (
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(data):
            written += os.write(descriptor, data[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(PENDING_EVIDENCE_ROOT, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return path


def qualify() -> dict[str, Any]:
    policy = _load_policy()
    authenticate_bootstrap(policy)
    source = authenticate_source(policy)
    with tempfile.TemporaryDirectory(
        prefix=f"{source.commit[:12]}-", dir=WORK_ROOT
    ) as temporary:
        temporary_path = Path(temporary)
        context_dir = temporary_path / "context"
        context_dir.mkdir(mode=0o700)
        context = construct_context(source, context_dir, policy=policy)
        authority = ProtectedDockerAuthority(
            policy, temporary_path / "docker-config"
        )
        daemon = authority.authenticate_daemon()
        image_id, result, execution = build_and_run(
            authority=authority,
            context_dir=context_dir,
            source=source,
            context=context,
        )
        evidence = {
            "schema": SCHEMA,
            "status": "controller_passed_pending_authority_teardown",
            "source": {"commit": source.commit, "tree": source.tree},
            "accepted_production": {
                "commit": ACCEPTED_PRODUCTION_SHA,
                "tree": ACCEPTED_PRODUCTION_TREE,
            },
            "platform": PLATFORM,
            "base_digest": BASE_DIGEST,
            "context": {
                "sha256": context.sha256,
                "recipe_sha256": context.recipe_sha256,
                "runner_sha256": context.runner_sha256,
                "file_count": context.file_count,
                "byte_count": context.byte_count,
            },
            "authority": {
                "runner_id": policy.runner_id,
                "git": policy.git.__dict__,
                "python": policy.python.__dict__,
                "docker": policy.docker.__dict__,
                "socket": policy.socket.__dict__,
                "daemon": daemon,
                "ephemeral_daemon_destroy_required": True,
            },
            "image": {"id": image_id, **execution},
            "result": result,
        }
        _safe_write_evidence(source.commit, evidence)
        return evidence


def _signal_handler(signum: int, _frame: Any) -> None:
    global _first_authority_interruption

    if not _authority_signal_latch.acquire(blocking=False):
        return
    _first_authority_interruption = AuthorityInterrupted(
        f"authority interrupted by signal {signum}"
    )
    if not _owned_command_active:
        raise _first_authority_interruption


def main() -> int:
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _signal_handler)
    try:
        evidence = qualify()
    except (QualificationRejected, QualificationUnavailable, AuthorityInterrupted) as exc:
        # This stdout is diagnostic only. A qualification exists only at the
        # fixed root-owned evidence path, written after joined cleanup.
        diagnostic = {
            "schema": SCHEMA,
            "status": "not_executed",
            "reason": str(exc),
        }
        print(json.dumps(diagnostic, sort_keys=True, separators=(",", ":")))
        return 2
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "controller_passed_pending_authority_teardown",
                "source": evidence["source"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
