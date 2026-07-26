"""Legacy/WatchPlan equivalence — the exit criterion for the ADR-008 API + runtime-compiler binding.

A ``WatchPlan`` that names only fields a legacy hand-authored :class:`~percept.adapters.Goal` already
had (a condition, an id, a source — no ``explicit`` overrides, no situation/profile/deployment
inputs beyond the SDK defaults) must compile to a BYTE-FOR-BYTE identical
:class:`~percept.adapters.WatchSpec` and an identical :attr:`~percept.adapters.Goal.state` as the
legacy path: hand-build a ``Goal`` and call ``add_goal`` directly. This is what makes the new
``ConfigResolver`` -> ``RuntimeCompiler`` -> ``apply_watch_plan`` boundary an ADDITIVE door onto the
existing runtime rather than a second, divergent one.

Two levels are checked:

1. Static: the compiled ``Goal`` (id/condition/say/source/state) and the ``WatchSpec`` built from it
   via :func:`percept.harness.tier0.compile_watchspec` match the legacy path's, field for field.
2. Dynamic: running BOTH goals through the real ``watch()`` seam over the SAME scripted, offline
   ``FakeVision`` fixture (zero API spend) yields identical :class:`~percept.contracts.WatchEvent`
   sequences.

No runtime execution beyond the offline scripted seam; no network, no API key required.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.contracts.watchplan import WatchPlan
from percept.goalsugar import goal_id_for
from percept.harness.tier0 import compile_watchspec
from percept.watch import watch
from percept.watchplan_apply import apply_watch_plan

_CONDITION = "a person is in the restricted zone"


def _watchspec_kwargs(core: Percept) -> dict:
    return dict(
        threshold=core.settings.gate.threshold,
        salience_stat=core.settings.edge.salience_stat,
        ego_compensate=core.settings.edge.ego_compensate,
        stat_thresholds={
            "mean": core.settings.edge.motion_gate,
            "block_max": core.settings.edge.motion_gate_block_max,
            "p95": core.settings.edge.motion_gate_p95,
        },
    )


class WatchPlanLegacyEquivalenceTests(unittest.TestCase):
    """``test_only_legacy_fields_produce_identical_watchspec_and_goal_state`` — the required
    equivalence test — plus a dynamic end-to-end companion over the same scripted fixture."""

    def test_only_legacy_fields_produce_identical_watchspec_and_goal_state(self):
        legacy_id = goal_id_for(_CONDITION)
        legacy_goal = Goal(id=legacy_id, condition=_CONDITION, say=_CONDITION, source="camera")

        core_legacy = Percept.create(mode="offline", discover_plugins=False)
        core_legacy.add_goal(legacy_goal)
        armed_legacy = core_legacy.goals.get(legacy_id)

        core_plan = Percept.create(mode="offline", discover_plugins=False)
        plan = WatchPlan(condition=_CONDITION)  # only-legacy fields: no id, no explicit, no situation
        receipt = apply_watch_plan(core_plan, plan)
        self.assertEqual(receipt.status, "applied")
        self.assertEqual(receipt.watch_ids, (legacy_id,))  # same deterministic id-minting scheme
        armed_plan = core_plan.goals.get(receipt.watch_ids[0])

        # -- Goal identity + state: byte-for-byte -----------------------------------------------
        self.assertEqual(armed_plan.id, armed_legacy.id)
        self.assertEqual(armed_plan.condition, armed_legacy.condition)
        self.assertEqual(armed_plan.say, armed_legacy.say)
        self.assertEqual(armed_plan.source, armed_legacy.source)
        self.assertEqual(armed_plan.modality, armed_legacy.modality)
        self.assertEqual(armed_plan.expiry, armed_legacy.expiry)
        self.assertEqual(armed_plan.origin, armed_legacy.origin)  # both STRUCTURED_TRUSTED via add_goal
        self.assertEqual(armed_plan.state, armed_legacy.state)

        # -- Settings the two cores resolved to: identical (both plain SDK defaults) ------------
        self.assertEqual(_watchspec_kwargs(core_plan), _watchspec_kwargs(core_legacy))

        # -- Compiled WatchSpec: byte-for-byte ---------------------------------------------------
        spec_legacy = compile_watchspec([armed_legacy], **_watchspec_kwargs(core_legacy))
        spec_plan = compile_watchspec([armed_plan], **_watchspec_kwargs(core_plan))
        self.assertEqual(spec_legacy, spec_plan)

    def test_watch_end_to_end_identical_events_legacy_vs_watchplan(self):
        """Same scripted fixture, same clock, two arming paths -> identical WatchEvent stream."""
        legacy_id = goal_id_for(_CONDITION)
        script = {"intruder": Verdict(satisfied=True, confidence=0.95, evidence="in zone")}

        core_legacy = Percept.create(
            vision=FakeVision(script=script), discover_plugins=False, clock=lambda: 100.0
        )
        core_plan = Percept.create(
            vision=FakeVision(script=script), discover_plugins=False, clock=lambda: 100.0
        )

        legacy_goal = Goal(id=legacy_id, condition=_CONDITION, say=_CONDITION, source="camera")
        plan = WatchPlan(condition=_CONDITION)

        async def _run():
            legacy_events = [
                ev async for ev in watch(["intruder"], legacy_goal, percept=core_legacy)
            ]
            plan_events = [ev async for ev in watch(["intruder"], plan, percept=core_plan)]
            return legacy_events, plan_events

        legacy_events, plan_events = asyncio.run(_run())

        self.assertEqual(len(legacy_events), 1)
        self.assertEqual(legacy_events, plan_events)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
