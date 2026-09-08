"""Persistence capability is rejected before a compiled Watch consumes or spends anything."""

from __future__ import annotations

import asyncio
import unittest

from percept import CompiledWatchSnapshotUnsupportedError, Goal, Percept
from percept.adapters import Verdict
from percept.contracts.watchplan import WatchPlan
from percept.edge.tier0 import gray_frame
from percept.persistence import InMemoryPersistence
from percept.runtime import run
from percept.watchplan_apply import apply_watch_plan


STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)


class CountingVision:
    def __init__(self) -> None:
        self.judge_calls = 0
        self.detect_calls = 0

    async def judge(self, condition, frame, **kwargs):
        self.judge_calls += 1
        return Verdict(False, 0.95, "clear")

    async def detect(self, frame, goals):
        self.detect_calls += 1
        return []

    async def detect_events(self, frames, goals):
        return []


class CountingSource:
    def __init__(self) -> None:
        self.pulls = 0

    def __iter__(self):
        for item in ((0.0, STILL, STILL), (1.0, MOVED, MOVED)):
            self.pulls += 1
            yield item


async def _drain(core, source, **kwargs):
    return [incident async for incident in run(core, source=source, **kwargs)]


def _compiled_core() -> tuple[Percept, CountingVision]:
    vision = CountingVision()
    core = Percept.create(vision=vision, discover_plugins=False)
    receipt = apply_watch_plan(core, WatchPlan(id="compiled", condition="a person is present"))
    if receipt.status not in ("applied", "applied_with_warnings"):
        raise AssertionError(f"compiled Watch did not arm: {receipt.status}")
    return core, vision


class CompiledWatchPersistencePreflightTests(unittest.TestCase):
    def test_every_persistence_mode_refuses_before_source_or_provider_work(self) -> None:
        cases = (
            ("final-save", {}),
            ("autosave", {"autosave_every_s": 0.001}),
            ("restore-empty-store", {"restore": True}),
        )
        for name, options in cases:
            with self.subTest(mode=name):
                core, vision = _compiled_core()
                source = CountingSource()
                store = InMemoryPersistence()
                health = []

                with self.assertRaisesRegex(
                    CompiledWatchSnapshotUnsupportedError,
                    "do not support compiled Watches",
                ):
                    asyncio.run(
                        _drain(
                            core,
                            source,
                            persistence=store,
                            on_health=health.append,
                            **options,
                        )
                    )

                self.assertEqual(source.pulls, 0)
                self.assertEqual(vision.judge_calls, 0)
                self.assertEqual(vision.detect_calls, 0)
                self.assertEqual(health, [], "no autosave error loop may start before rejection")
                self.assertEqual(store._store, {}, "preflight must not write a partial artifact")
                self.assertIn("compiled", core._watch_runtime_bindings)

    def test_plain_goal_persistence_still_consumes_and_saves_normally(self) -> None:
        vision = CountingVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal(id="plain", condition="a person is present", say="present"))
        source = CountingSource()
        store = InMemoryPersistence()

        incidents = asyncio.run(_drain(core, source, persistence=store, judge_dispatch="inline"))

        self.assertEqual(incidents, [])
        self.assertEqual(source.pulls, 2)
        self.assertGreaterEqual(vision.judge_calls, 1)
        self.assertIsNotNone(store.load("percept"))
        self.assertEqual(store.load("percept_delivery_ledger"), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
