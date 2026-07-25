"""Validate the installed public perception runtime without a source checkout.

Run this file with the wheel environment's Python from a directory outside the
repository.  It exercises the public programmable Watch API over the bundled
deterministic replay, rather than importing implementation modules directly.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path


_CREDENTIALS = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_AISTUDIO_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
    "DEEPGRAM_API_KEY",
)


def _assert_installed_distribution(percept: object) -> importlib.metadata.Distribution:
    distribution = importlib.metadata.distribution("percept-vision")
    installed_root = Path(distribution.locate_file("percept")).resolve()
    imported_root = Path(getattr(percept, "__file__")).resolve().parent
    if imported_root != installed_root:
        raise AssertionError(
            "import percept did not resolve to the installed percept-vision distribution: "
            f"imported {imported_root}, distribution owns {installed_root}"
        )
    if (imported_root.parent / "pyproject.toml").is_file():
        raise AssertionError(f"import percept resolved to a source checkout: {imported_root}")
    return distribution


def _assert_public_exports(percept: object) -> None:
    missing = [name for name in percept.__all__ if not hasattr(percept, name)]
    if missing:
        raise AssertionError(f"public percept exports are not selectable: {missing}")


def _assert_declared_entry_points(distribution: importlib.metadata.Distribution) -> None:
    entry_points = tuple(distribution.entry_points)
    if not entry_points:
        raise AssertionError("percept-vision declares no installable entry points")
    for entry_point in entry_points:
        target = entry_point.load()
        if not callable(target):
            raise AssertionError(
                f"declared entry point {entry_point.group}:{entry_point.name} is not callable"
            )
        if entry_point.group == "console_scripts":
            script = Path(sys.executable).parent / entry_point.name
            if not script.exists():
                raise AssertionError(f"declared console entry point was not installed: {entry_point.name}")


async def _watch_fixture(percept: object) -> None:
    fixture = percept.load_fixture()
    events = [
        event
        async for event in percept.watch(
            percept.offline_frames(fixture),
            percept.offline_goals(fixture),
            percept=percept.offline_core(fixture),
        )
    ]
    expected = [(item["goal_id"], item["action"]) for item in fixture.expect]
    observed = [(event.goal_id, event.action) for event in events]
    if observed != expected:
        raise AssertionError(f"offline Watch events differ from its bundled manifest: {observed} != {expected}")
    if not events:
        raise AssertionError("bundled Watch manifest declares no event")
    for event in events:
        if event.verdict is None or "NOT a perception result" not in event.verdict.evidence:
            raise AssertionError("offline Watch event lacks replay provenance")


def main() -> int:
    for name in _CREDENTIALS:
        os.environ.pop(name, None)

    import percept

    distribution = _assert_installed_distribution(percept)
    _assert_public_exports(percept)
    _assert_declared_entry_points(distribution)
    if importlib.util.find_spec("eval") is not None:
        raise AssertionError("the repository-only eval package is selectable from the installed wheel")
    asyncio.run(_watch_fixture(percept))
    print(f"installed public perception runtime OK: {percept.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
