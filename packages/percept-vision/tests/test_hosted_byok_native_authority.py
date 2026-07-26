from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


MODULE = (
    Path(__file__).parents[1]
    / "tools"
    / "qualify_hosted_byok_native.py"
)
spec = importlib.util.spec_from_file_location("hosted_byok_native_authority", MODULE)
assert spec and spec.loader
authority = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = authority
spec.loader.exec_module(authority)


def _reset_authority_signal_state() -> None:
    authority._authority_signal_latch = authority.threading.Lock()
    authority._first_authority_interruption = None
    authority._owned_command_active = False


def _identity(path: Path, *, socket: bool = False) -> object:
    return authority.FileIdentity(
        path=str(path),
        sha256="" if socket else "a" * 64,
        uid=0,
        gid=0,
        mode=0o600 if socket else 0o755,
        device=1,
        inode=2,
    )


def _policy(*, commit: str = "b" * 40, tree: str = "c" * 40) -> object:
    return authority.AuthorityPolicy(
        qualification_commit=commit,
        qualification_tree=tree,
        controller_sha256="d" * 64,
        recipe_sha256="e" * 64,
        runner_sha256="f" * 64,
        git=_identity(authority.FIXED_GIT),
        python=_identity(authority.FIXED_PYTHON),
        docker=_identity(authority.FIXED_DOCKER),
        socket=_identity(authority.FIXED_DOCKER_SOCKET, socket=True),
        daemon_id="daemon-id",
        daemon_version="1.2.3",
        docker_root_dir="/protected/docker",
        runner_id="runner-id",
    )


def _policy_json() -> dict:
    policy = _policy()
    return {
        "schema": 1,
        "qualification": {
            "commit": policy.qualification_commit,
            "tree": policy.qualification_tree,
            "controller_sha256": policy.controller_sha256,
            "recipe_sha256": policy.recipe_sha256,
            "runner_sha256": policy.runner_sha256,
        },
        "accepted_production": {
            "commit": authority.ACCEPTED_PRODUCTION_SHA,
            "tree": authority.ACCEPTED_PRODUCTION_TREE,
        },
        "tools": {
            "git": policy.git.__dict__,
            "python": policy.python.__dict__,
            "docker": policy.docker.__dict__,
            "socket": policy.socket.__dict__,
        },
        "daemon": {
            "id": policy.daemon_id,
            "version": policy.daemon_version,
            "docker_root_dir": policy.docker_root_dir,
        },
        "authority": {
            "runner_id": policy.runner_id,
            "ephemeral_daemon": True,
            "destroy_on_every_outcome": True,
        },
    }


def _source() -> object:
    return authority.SourceIdentity(commit="b" * 40, tree="c" * 40)


def _context() -> object:
    return authority.ContextIdentity(
        sha256="1" * 64,
        recipe_sha256="2" * 64,
        runner_sha256="3" * 64,
        file_count=195,
        byte_count=4_000_000,
    )


def _native_result() -> dict:
    observations = []
    for case in authority.CASES:
        provider, scenario = case.split(":")
        observations.append(
            {
                "provider": provider,
                "scenario": scenario,
                "outcome": {
                    "success": "success",
                    "deadline": "deadline",
                    "cancel": "cancelled",
                }[scenario],
                "provider_called": True,
                "process_exit_observed": True,
                "pids_after_return": [],
                "pids_after_horizon": [],
                "observed_horizon_ms": 2750,
                "late_effect_paths": [],
            }
        )
    return {
        "schema": 3,
        "status": "passed",
        "platform": "linux",
        "architecture": "amd64",
        "python": "3.12",
        "cases": list(authority.CASES),
        "observations": observations,
        "environment_poisoned": True,
        "no_descendants_observed": True,
        "joined_observed": True,
        "late_effect_observed": False,
    }


def _image(labels: dict[str, str]) -> list[dict]:
    return [
        {
            "Id": "sha256:" + "4" * 64,
            "Os": "linux",
            "Architecture": "amd64",
            "Config": {
                "Labels": dict(labels),
                "Entrypoint": [
                    "/usr/local/bin/python",
                    "-I",
                    "/qualification/hosted_byok_native_matrix.py",
                ],
                "Cmd": None,
                "WorkingDir": "/qualification",
            },
        }
    ]


def _container() -> list[dict]:
    return [
        {
            "Id": "5" * 64,
            "Image": "sha256:" + "4" * 64,
            "Name": "/percept-test",
            "Config": {"User": "65534:65534"},
            "HostConfig": {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "PidsLimit": 128,
                "Memory": 768 * 1024 * 1024,
                "NanoCpus": 2_000_000_000,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "Tmpfs": {
                    "/tmp": "rw,noexec,nosuid,nodev,size=134217728"
                },
            },
        }
    ]


def test_policy_is_closed_and_requires_external_ephemeral_authority() -> None:
    assert authority.parse_policy(_policy_json()) == _policy()
    for mutation in (
        lambda value: value.update(extra=True),
        lambda value: value["authority"].update(ephemeral_daemon=False),
        lambda value: value["authority"].update(destroy_on_every_outcome=False),
        lambda value: value["qualification"].update(commit="9" * 39),
        lambda value: value["tools"]["socket"].update(
            path="/tmp/caller-docker.sock"
        ),
    ):
        value = _policy_json()
        mutation(value)
        with pytest.raises(authority.QualificationRejected):
            authority.parse_policy(value)


def test_controller_has_no_cli_source_result_tool_or_evidence_inputs() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "argparse" not in source
    assert "--source" not in source
    assert "--result" not in source
    assert "--docker" not in source
    assert "--evidence" not in source
    assert authority.POLICY_PATH.is_absolute()
    assert authority.PENDING_EVIDENCE_ROOT.is_absolute()
    assert authority.WORK_ROOT.is_absolute()


def test_direct_checkout_execution_with_hostile_pythonpath_is_not_executed(
    tmp_path: Path,
) -> None:
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    marker = tmp_path / "imported"
    (hostile / "selectors.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "raise RuntimeError('hostile import')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(hostile)
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", str(MODULE)],
        cwd="/",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert not marker.exists()
    diagnostic = json.loads(completed.stdout)
    assert diagnostic["status"] == "not_executed"


def test_future_self_print_commit_is_rejected_by_external_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    future = "9" * 40
    values = {
        (
            "rev-parse",
            "--verify",
            f"{policy.qualification_commit}^{{commit}}",
        ): future,
        (
            "rev-parse",
            "--verify",
            f"{policy.qualification_commit}^{{tree}}",
        ): policy.qualification_tree,
        (
            "rev-parse",
            "--verify",
            f"{policy.qualification_commit}^",
        ): authority.ACCEPTED_PRODUCTION_SHA,
        (
            "rev-parse",
            "--verify",
            f"{authority.ACCEPTED_PRODUCTION_SHA}^{{tree}}",
        ): authority.ACCEPTED_PRODUCTION_TREE,
    }
    monkeypatch.setattr(
        authority, "_git_text", lambda args, *, policy: values[tuple(args)]
    )
    with pytest.raises(
        authority.QualificationRejected, match="Git object is not exact"
    ):
        authority.authenticate_source(policy)


def test_exact_native_result_passes() -> None:
    assert authority.validate_native_result(_native_result()) == _native_result()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(cases=value["cases"][:-1]),
        lambda value: value.update(
            cases=[*value["cases"][:-1], value["cases"][0]]
        ),
        lambda value: value.update(
            cases=[*value["cases"][:-1], ["forged-unhashable"]]
        ),
        lambda value: value["observations"].pop(),
        lambda value: value["observations"][0].update(
            process_exit_observed=False
        ),
        lambda value: value["observations"][0].update(
            pids_after_return=[22]
        ),
        lambda value: value["observations"][0].update(
            pids_after_horizon=[23]
        ),
        lambda value: value["observations"][0].update(
            observed_horizon_ms=2499
        ),
        lambda value: value["observations"][0].update(
            late_effect_paths=["gemini-deadline-late-effect"]
        ),
        lambda value: value.update(platform="darwin"),
        lambda value: value.update(architecture="arm64"),
        lambda value: value.update(joined_observed=False),
        lambda value: value.update(forged=True),
    ],
)
def test_skipped_duplicate_forged_or_nonquiescent_result_is_rejected(
    mutation,
) -> None:
    value = _native_result()
    mutation(value)
    with pytest.raises(authority.QualificationRejected):
        authority.validate_native_result(value)


def test_duplicate_json_keys_and_nonfinite_constants_are_rejected() -> None:
    with pytest.raises(authority.QualificationRejected, match="duplicate JSON"):
        authority._strict_json(b'{"schema":3,"schema":3}')
    with pytest.raises(authority.QualificationRejected, match="invalid JSON"):
        authority._strict_json(b'{"value":NaN}')


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value[0].update(Os="darwin"),
        lambda value: value[0].update(Architecture="arm64"),
        lambda value: value[0]["Config"].update(Entrypoint=["/bin/sh"]),
        lambda value: value[0]["Config"]["Labels"].update(
            {"org.percept.hosted-byok.base-digest": "sha256:" + "0" * 64}
        ),
        lambda value: value[0]["Config"]["Labels"].update(
            {"org.percept.hosted-byok.platform": "linux/arm64"}
        ),
    ],
)
def test_wrong_image_labels_base_platform_or_entrypoint_is_rejected(
    mutation,
) -> None:
    labels = authority.expected_labels(_source(), _context())
    value = _image(labels)
    mutation(value)
    with pytest.raises(authority.QualificationRejected):
        authority.validate_image(
            value, image_id="sha256:" + "4" * 64, labels=labels
        )


def test_container_requires_exact_id_image_name_and_resource_boundary() -> None:
    authority.validate_container(
        _container(),
        container_id="5" * 64,
        image_id="sha256:" + "4" * 64,
        name="percept-test",
    )
    for mutation in (
        lambda value: value[0]["HostConfig"].update(NetworkMode="host"),
        lambda value: value[0]["HostConfig"].update(ReadonlyRootfs=False),
        lambda value: value[0]["HostConfig"].update(CapDrop=[]),
        lambda value: value[0]["Config"].update(User="0"),
    ):
        value = _container()
        mutation(value)
        with pytest.raises(authority.QualificationRejected):
            authority.validate_container(
                value,
                container_id="5" * 64,
                image_id="sha256:" + "4" * 64,
                name="percept-test",
            )


def test_group_writable_or_symlinked_tool_and_socket_paths_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = tmp_path / "tool"
    tool.write_text("tool", encoding="utf-8")
    tool.chmod(0o775)
    monkeypatch.setattr(authority, "_path_components", lambda path: (path,))
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    with pytest.raises(authority.QualificationUnavailable, match="group/world"):
        authority.protected_identity(tool, kind="file")

    link = tmp_path / "link"
    link.symlink_to(tool)
    with pytest.raises(authority.QualificationUnavailable, match="symlink"):
        authority.protected_identity(link, kind="file")


def test_timeout_terminates_and_joins_entire_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "late-child"
    child = (
        "import time\n"
        "from pathlib import Path\n"
        "time.sleep(0.5)\n"
        f"Path({str(marker)!r}).write_text('late')\n"
    )
    parent = (
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable,'-c',{child!r}])\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    with pytest.raises(authority.QualificationUnavailable, match="timed out"):
        authority._read_streams_bounded(
            process, authority.Command(tuple(), timeout_s=0.05)
        )
    time.sleep(0.7)
    assert process.poll() is not None
    assert not marker.exists()


def test_signal_between_process_creation_and_popen_return_is_deferred_and_joined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_popen = subprocess.Popen
    captured: list[subprocess.Popen[bytes]] = []

    def boundary_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        captured.append(process)
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGHUP)
        return process

    previous_term = signal.getsignal(signal.SIGTERM)
    previous_hup = signal.getsignal(signal.SIGHUP)
    fallback_leader_cleanup = False
    fallback_group_cleanup = False
    _reset_authority_signal_state()
    monkeypatch.setattr(authority.subprocess, "Popen", boundary_popen)
    try:
        signal.signal(signal.SIGTERM, authority._signal_handler)
        signal.signal(signal.SIGHUP, authority._signal_handler)
        with pytest.raises(authority.AuthorityInterrupted) as interrupted:
            authority.run_command(
                authority.Command(
                    (sys.executable, "-c", "import time; time.sleep(60)"),
                    timeout_s=5.0,
                ),
                cwd=tmp_path,
                env=os.environ,
            )
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGHUP, previous_hup)
        if captured:
            process = captured[0]
            fallback_leader_cleanup = process.poll() is None
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                pass
            else:
                fallback_group_cleanup = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.wait()
        _reset_authority_signal_state()

    assert interrupted.value.args == (
        f"authority interrupted by signal {signal.SIGTERM}",
    )
    assert len(captured) == 1
    process = captured[0]
    assert not fallback_leader_cleanup
    assert not fallback_group_cleanup
    assert process.poll() is not None
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


def test_catchable_signal_joins_real_leader_and_descendant_and_preserves_error(
    tmp_path: Path,
) -> None:
    descendant_pid_path = tmp_path / "descendant-pid"
    late_marker = tmp_path / "late-descendant"
    descendant = (
        "import signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(2.5)\n"
        f"Path({str(late_marker)!r}).write_text('survived')\n"
        "time.sleep(60)\n"
    )
    leader = (
        "import signal,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"child=subprocess.Popen([sys.executable,'-c',{descendant!r}])\n"
        f"Path({str(descendant_pid_path)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", leader],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + 2.0
    while not descendant_pid_path.exists():
        if time.monotonic() >= deadline:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise AssertionError("real descendant did not start")
        time.sleep(0.01)
    descendant_pid = int(descendant_pid_path.read_text(encoding="ascii"))

    previous = signal.getsignal(signal.SIGTERM)
    senders = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import os,signal,time;"
                    f"time.sleep({delay!r});"
                    f"os.kill({os.getpid()},signal.SIGTERM)"
                ),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for delay in (0.05, 0.1)
    ]
    fallback_leader_cleanup = False
    fallback_group_cleanup = False
    _reset_authority_signal_state()
    try:
        signal.signal(signal.SIGTERM, authority._signal_handler)
        with pytest.raises(authority.AuthorityInterrupted) as interrupted:
            authority._read_streams_bounded(
                process, authority.Command(tuple(), timeout_s=5.0)
            )
    finally:
        for sender in senders:
            sender.wait(timeout=2)
        signal.signal(signal.SIGTERM, previous)
        fallback_leader_cleanup = process.poll() is None
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            fallback_group_cleanup = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.wait()
        _reset_authority_signal_state()

    assert interrupted.value.args == (
        f"authority interrupted by signal {signal.SIGTERM}",
    )
    assert not fallback_leader_cleanup
    assert not fallback_group_cleanup
    assert process.poll() is not None
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)

    deadline = time.monotonic() + 1.0
    while True:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            raise AssertionError("real descendant survived process-group join")
        time.sleep(0.01)
    assert not late_marker.exists()


class _FakeDocker:
    def __init__(self, labels: dict[str, str], *, fail_start: bool = False):
        self.labels = labels
        self.fail_start = fail_start
        self.container_removed = False
        self.image_removed = False
        self.calls: list[tuple[str, ...]] = []

    def run(self, args, *, cwd, timeout_s, max_output=authority.MAX_OUTPUT_BYTES):
        del timeout_s, max_output
        args = tuple(args)
        self.calls.append(args)
        if args[0] == "build":
            iidfile = Path(args[args.index("--iidfile") + 1])
            iidfile.write_text("sha256:" + "4" * 64, encoding="ascii")
            return authority.Completed(0, b"", b"")
        if args[:2] == ("image", "inspect"):
            if self.image_removed:
                return authority.Completed(1, b"", b"absent")
            return authority.Completed(
                0,
                json.dumps(_image(self.labels)).encode(),
                b"",
            )
        if args[:2] == ("container", "create"):
            return authority.Completed(0, ("5" * 64 + "\n").encode(), b"")
        if args[:2] == ("container", "inspect") and "--format" not in args:
            if self.container_removed:
                return authority.Completed(1, b"", b"absent")
            name = args[args.index("inspect") + 1]
            assert name == "5" * 64
            value = _container()
            generated_name = next(
                call[call.index("--name") + 1]
                for call in self.calls
                if call[:2] == ("container", "create")
            )
            value[0]["Name"] = f"/{generated_name}"
            return authority.Completed(0, json.dumps(value).encode(), b"")
        if args[:2] == ("container", "start"):
            if self.fail_start:
                raise authority.AuthorityInterrupted("simulated signal")
            return authority.Completed(
                0, (json.dumps(_native_result()) + "\n").encode(), b""
            )
        if args[:2] == ("container", "inspect") and "--format" in args:
            return authority.Completed(
                0,
                json.dumps(
                    {
                        "Running": False,
                        "ExitCode": 0,
                        "FinishedAt": "2026-07-26T00:00:00Z",
                    }
                ).encode(),
                b"",
            )
        if args[:3] == ("container", "rm", "--force"):
            self.container_removed = True
            return authority.Completed(0, b"", b"")
        if args[:3] == ("image", "rm", "--force"):
            self.image_removed = True
            return authority.Completed(0, b"", b"")
        raise AssertionError((args, cwd))

    def cleanup_container(self, reference, cwd):
        del reference
        self.run(
            ("container", "rm", "--force", "5" * 64),
            cwd=cwd,
            timeout_s=30,
        )
        result = self.run(
            ("container", "inspect", "5" * 64),
            cwd=cwd,
            timeout_s=15,
        )
        assert result.returncode != 0

    def cleanup_image(self, reference, cwd):
        del reference
        self.run(
            ("image", "rm", "--force", "sha256:" + "4" * 64),
            cwd=cwd,
            timeout_s=30,
        )
        result = self.run(
            ("image", "inspect", "sha256:" + "4" * 64),
            cwd=cwd,
            timeout_s=15,
        )
        assert result.returncode != 0


def test_success_uses_named_cid_execution_and_proves_joined_cleanup(
    tmp_path: Path,
) -> None:
    labels = authority.expected_labels(_source(), _context())
    docker = _FakeDocker(labels)
    image_id, result, execution = authority.build_and_run(
        authority=docker,
        context_dir=tmp_path,
        source=_source(),
        context=_context(),
    )
    assert image_id == "sha256:" + "4" * 64
    assert result == _native_result()
    assert execution["container_id"] == "5" * 64
    assert docker.container_removed and docker.image_removed
    create = next(
        call for call in docker.calls if call[:2] == ("container", "create")
    )
    assert "--name" in create
    start = next(
        call for call in docker.calls if call[:2] == ("container", "start")
    )
    assert start[-1] == "5" * 64


def test_signal_after_engine_create_still_removes_exact_container_and_image(
    tmp_path: Path,
) -> None:
    docker = _FakeDocker(
        authority.expected_labels(_source(), _context()),
        fail_start=True,
    )
    with pytest.raises(authority.AuthorityInterrupted):
        authority.build_and_run(
            authority=docker,
            context_dir=tmp_path,
            source=_source(),
            context=_context(),
        )
    assert docker.container_removed and docker.image_removed


def test_cli_killed_with_unknown_engine_build_never_returns_result_and_attempts_cleanup(
    tmp_path: Path,
) -> None:
    class BuildTimeout:
        def __init__(self) -> None:
            self.cleaned_container: list[str] = []
            self.cleaned_image: list[str] = []

        def run(self, args, *, cwd, timeout_s, max_output=authority.MAX_OUTPUT_BYTES):
            del cwd, timeout_s, max_output
            if args[0] == "build":
                raise authority.QualificationUnavailable(
                    "protected command timed out"
                )
            raise AssertionError(args)

        def cleanup_container(self, reference, cwd):
            del cwd
            self.cleaned_container.append(reference)

        def cleanup_image(self, reference, cwd):
            del cwd
            self.cleaned_image.append(reference)

    docker = BuildTimeout()
    with pytest.raises(authority.QualificationUnavailable, match="timed out"):
        authority.build_and_run(
            authority=docker,
            context_dir=tmp_path,
            source=_source(),
            context=_context(),
        )
    assert len(docker.cleaned_container) == 1
    assert docker.cleaned_container[0].startswith(
        "percept-hosted-byok-native-"
    )
    assert len(docker.cleaned_image) == 1
    assert docker.cleaned_image[0].startswith("percept-hosted-byok-native:")


def test_runner_derives_quiescence_fields_and_waits_beyond_late_effect() -> None:
    runner = (
        Path(__file__).parents[1]
        / "tools"
        / "hosted_byok_native_matrix.py"
    ).read_text(encoding="utf-8")
    assert "_LATE_EFFECT_HORIZON_S = 2.75" in runner
    assert '"no_descendants_observed": joined' in runner
    assert '"joined_observed": joined' in runner
    assert '"late_effect_observed": not no_late_effect' in runner
    assert '"no_descendants_observed": True' not in runner
    assert '"joined_observed": True' not in runner


def test_recipe_is_exact_digest_pinned_networkless_and_exec_form() -> None:
    recipe = (
        Path(__file__).parents[1] / "Dockerfile.hosted-byok-native"
    ).read_text(encoding="utf-8")
    assert recipe == authority.EXPECTED_DOCKERFILE
    assert f"python:3.12-slim@{authority.BASE_DIGEST}" in recipe
    assert "\nRUN " not in recipe
    assert "pip " not in recipe.lower()
    assert 'ENTRYPOINT ["/usr/local/bin/python", "-I"' in recipe


def test_evidence_uses_exclusive_nofollow_write_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authority, "PENDING_EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr(
        authority,
        "protected_identity",
        lambda path, *, kind: authority.FileIdentity(
            str(path), "", 0, 0, 0o700, 1, 1
        ),
    )
    path = authority._safe_write_evidence("b" * 40, {"status": "passed"})
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        authority._safe_write_evidence("b" * 40, {"status": "forged"})


def test_documentation_forbids_shared_daemon_and_local_native_claim() -> None:
    document = (
        Path(__file__).parents[1]
        / "HOSTED_BYOK_NATIVE_QUALIFICATION.md"
    ).read_text(encoding="utf-8")
    assert "not a developer-side root launcher" in document
    assert "ephemeral Docker daemon" in document
    assert "SIGKILL" in document
    assert "No final external envelope means no native qualification" in document
