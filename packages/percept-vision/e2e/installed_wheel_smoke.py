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
                raise AssertionError(
                    f"declared console entry point was not installed: {entry_point.name}"
                )


def _assert_free_pack_surface(distribution: importlib.metadata.Distribution) -> None:
    from percept.domain_pack import (
        DomainPack,
        DomainPackError,
        DomainPackRegistry,
        get_domain_pack,
        watch_template,
    )

    official = {
        "physical_scene_starter@1": {
            "door_left_open",
            "package_appeared",
            "standing_water",
            "zone_occupied",
        },
        "ego_motion_starter@1": {
            "camera_view_moving",
            "scene_transition",
            "object_focus_gained",
            "task_step_reached",
        },
    }
    for qualified_name, expected_templates in official.items():
        pack = get_domain_pack(qualified_name)
        if set(pack.templates) != expected_templates:
            raise AssertionError(
                f"installed {qualified_name} templates differ: {sorted(pack.templates)}"
            )
        if pack.fixtures or pack.quality_bars:
            raise AssertionError(
                f"unqualified installed starter {qualified_name} exposes quality claims"
            )
        plan = watch_template(f"{qualified_name}/{sorted(expected_templates)[0]}")
        if not plan.condition:
            raise AssertionError(f"installed {qualified_name} did not render a WatchPlan")

    # The original public name remains resolvable; the official starter is additive.
    get_domain_pack("facility_observation@1")

    discovered: set[str] = set()
    for entry_point in distribution.entry_points:
        if entry_point.group != "percept.packs":
            continue
        registry = DomainPackRegistry()
        target = entry_point.load()
        target(registry.register)
        target(registry.register)
        discovered.update(registry.names())

        installed = registry.all()[0]
        conflict_registry = DomainPackRegistry()
        conflict = DomainPack(
            id=installed.id,
            version=installed.version,
            notes="installed-wheel collision sentinel",
        )
        conflict_registry.register(conflict)
        try:
            target(conflict_registry.register)
        except DomainPackError:
            pass
        else:
            raise AssertionError(
                f"installed percept.packs entry point silently replaced {installed.qualified_name}"
            )
        if conflict_registry.resolve(installed.qualified_name) is not conflict:
            raise AssertionError(
                f"installed percept.packs entry point changed {installed.qualified_name} on conflict"
            )
    expected = {*official, "facility_observation@1"}
    if not expected.issubset(discovered):
        raise AssertionError(
            f"installed percept.packs entry points missing {sorted(expected - discovered)}"
        )


async def _assert_transition_ordering(percept: object) -> None:
    from percept.adapters import Verdict
    from percept.domain_pack import get_domain_pack, watch_template
    from percept.watchplan_apply import apply_watch_plan

    plan = watch_template("ego_motion_starter@1/scene_transition")
    before = plan.explicit["Goal.state.from"]
    after = plan.explicit["Goal.state.to"]

    class ConditionVision:
        async def judge(self, condition, frame, **_kwargs):
            token = str(frame)
            if token == "both":
                return Verdict(True, 0.99, "both predicates true on one observation")
            if condition == before:
                return Verdict(token == "from", 0.99, f"from={token == 'from'}")
            if condition == after:
                return Verdict(token == "to", 0.99, f"to={token == 'to'}")
            raise AssertionError(f"unexpected condition: {condition!r}")

        async def detect(self, frame, goals):
            return []

        async def detect_events(self, frames, goals):
            return []

    core = percept.Percept.create(vision=ConditionVision(), discover_plugins=False)
    receipt = apply_watch_plan(
        core,
        plan,
        domain_pack=get_domain_pack("ego_motion_starter@1"),
    )
    if receipt.status not in {"applied", "applied_with_warnings"}:
        raise AssertionError(f"installed transition Watch did not apply: {receipt.status}")

    actions: list[str] = []
    for index, token in enumerate(
        ("both", "to", "from", "to", "both", "to", "from", "to"),
        start=1,
    ):
        actions.extend(
            event.action for event in await core.perceive_judged(token, t=float(index))
        )
    if actions != ["fire", "fire"]:
        raise AssertionError(
            "installed transition did not require a later exclusive observation: "
            f"{actions}"
        )


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
        raise AssertionError(
            f"offline Watch events differ from its bundled manifest: {observed} != {expected}"
        )
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
    _assert_free_pack_surface(distribution)
    if importlib.util.find_spec("eval") is not None:
        raise AssertionError(
            "the repository-only eval package is selectable from the installed wheel"
        )
    asyncio.run(_assert_transition_ordering(percept))
    asyncio.run(_watch_fixture(percept))
    print(f"installed public perception runtime OK: {percept.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
