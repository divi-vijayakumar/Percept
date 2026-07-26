"""Run the six Hosted BYOK cases and prove process quiescence in-container.

This module runs only in the digest-pinned Linux/amd64 qualification image.
It uses the exact reviewed fake-transport composition helpers from
``test_hosted_byok.py``.  A case is successful only after every other PID in
the private container PID namespace is gone and the deliberate late-effect
horizon has elapsed without a file effect.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import textwrap
import time


CASES = tuple(
    (provider, scenario)
    for provider in ("gemini", "openrouter")
    for scenario in ("success", "deadline", "cancel")
)
_HARNESS_SCENARIO = {
    "success": "success",
    "deadline": "timeout",
    "cancel": "cancel",
}
_SOURCE = Path("/qualification/tests/test_hosted_byok.py")
_WANTED = {
    "_HOSTED_ENV_KEYS",
    "_HOSTILE_STARTUP_ENV_KEYS",
    "_write_production_transport_fakes",
    "_run_production_composition_harness",
}
_LATE_EFFECT_HORIZON_S = 2.75
_QUIESCENCE_TIMEOUT_S = 5.0


def _reviewed_helpers() -> tuple[object, frozenset[str]]:
    source = _SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_SOURCE))
    selected = [
        node
        for node in tree.body
        if getattr(node, "name", None) in _WANTED
        or any(
            getattr(target, "id", None) in _WANTED
            for target in getattr(node, "targets", ())
        )
    ]
    selected_names = {
        getattr(node, "name", None)
        for node in selected
        if getattr(node, "name", None) is not None
    }
    selected_names.update(
        getattr(target, "id", None)
        for node in selected
        for target in getattr(node, "targets", ())
    )
    if selected_names != _WANTED:
        raise SystemExit("reviewed Hosted BYOK helper set is incomplete")
    namespace = {
        "__file__": str(_SOURCE),
        "Path": Path,
        "asyncio": asyncio,
        "json": json,
        "os": os,
        "subprocess": subprocess,
        "sys": sys,
        "textwrap": textwrap,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(_SOURCE), "exec"),
        namespace,
    )
    hostile = namespace["_HOSTILE_STARTUP_ENV_KEYS"]
    if type(hostile) is not set or not hostile:
        raise SystemExit("hostile environment set is not exact")
    return namespace["_run_production_composition_harness"], frozenset(hostile)


def _other_pids() -> tuple[int, ...]:
    self_pid = os.getpid()
    try:
        values = tuple(
            sorted(
                int(entry.name)
                for entry in Path("/proc").iterdir()
                if entry.name.isdecimal() and int(entry.name) != self_pid
            )
        )
    except OSError as exc:
        raise SystemExit("private Linux /proc is unavailable") from exc
    return values


def _wait_quiescent() -> tuple[int, ...]:
    deadline = time.monotonic() + _QUIESCENCE_TIMEOUT_S
    while True:
        pids = _other_pids()
        if not pids:
            return pids
        if time.monotonic() >= deadline:
            raise SystemExit(f"native case retained descendants: {pids!r}")
        time.sleep(0.025)


def _effect_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(path.relative_to(root))
            for path in root.glob("*-late-effect")
            if path.is_file()
        )
    )


def main() -> int:
    if sys.platform != "linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise SystemExit("Hosted BYOK native matrix requires Linux x86_64")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Hosted BYOK native matrix requires Python 3.12")
    if os.getpid() != 1:
        raise SystemExit("native matrix must own the private container PID namespace")

    run_case, hostile = _reviewed_helpers()
    observations: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="hosted-byok-native-") as temporary:
        root = Path(temporary)
        for provider, scenario in CASES:
            result = run_case(
                root,
                provider=provider,
                scenario=_HARNESS_SCENARIO[scenario],
                python_executable="/usr/local/bin/python",
            )
            expected_outcome = {
                "success": "success",
                "deadline": "deadline",
                "cancel": "cancelled",
            }[scenario]
            if result != {
                "outcome": expected_outcome,
                "provider_called": True,
                "late_effect": False,
            }:
                raise SystemExit(f"native matrix case failed: {provider}:{scenario}")

            pids_after_return = _wait_quiescent()
            before_horizon = time.monotonic()
            time.sleep(_LATE_EFFECT_HORIZON_S)
            observed_horizon_s = time.monotonic() - before_horizon
            pids_after_horizon = _wait_quiescent()
            late_effect_paths = _effect_paths(root)
            if late_effect_paths:
                raise SystemExit(
                    f"native case produced a late effect: {provider}:{scenario}"
                )
            observations.append(
                {
                    "provider": provider,
                    "scenario": scenario,
                    "outcome": expected_outcome,
                    "provider_called": True,
                    "process_exit_observed": True,
                    "pids_after_return": list(pids_after_return),
                    "pids_after_horizon": list(pids_after_horizon),
                    "observed_horizon_ms": int(observed_horizon_s * 1000),
                    "late_effect_paths": list(late_effect_paths),
                }
            )

    joined = all(
        observation["process_exit_observed"]
        and not observation["pids_after_return"]
        and not observation["pids_after_horizon"]
        for observation in observations
    )
    no_late_effect = all(
        not observation["late_effect_paths"] for observation in observations
    )
    output = {
        "schema": 3,
        "status": "passed",
        "platform": "linux",
        "architecture": "amd64",
        "python": "3.12",
        "cases": [f"{provider}:{scenario}" for provider, scenario in CASES],
        "observations": observations,
        "environment_poisoned": bool(hostile),
        "no_descendants_observed": joined,
        "joined_observed": joined,
        "late_effect_observed": not no_late_effect,
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
