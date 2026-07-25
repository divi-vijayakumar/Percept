"""P06-001 prepared ``apply_watch_plan`` with bounded rollback.

Phase 1 PREPARE builds every goal + every judge binding with NO runtime mutation. Cache invalidation
is preflighted before Phase 2 COMMIT, and owned goal/binding/ledger state is restored if commit fails.
A judge PREPARE failure must leave the runtime untouched and surface a structured rejected receipt.

All doubles are LOCAL. ``percept.judge_registry`` (built by the parallel wave) is not importable in
this worktree, so the tests install a fake module matching its pinned interface
(``get_or_create_adapter(spec, cfg)`` + a ``BoundJudge`` frozen dataclass) into ``sys.modules`` for
the duration of the call — exactly the module ``watchplan_apply`` lazily imports.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.contracts.watchplan import ResolvedJudgeSpec, WatchPlan
from percept.deployment_judges import DeploymentJudges
from percept.goals import GoalOp
from percept.harness.tier0 import gray_frame
from percept.resolver import ConfigResolver
from percept.runtime.worker import RuntimeStats, run
from percept.sourceprofile import SourceProfile
from percept.watchplan_apply import apply_watch_plan


class SpyJudge:
    """A minimal judge/vision double — records nothing beyond existing, returns a scripted verdict."""

    PARSES_CLAIM_CHANNEL = True

    def __init__(self, name: str = "spy") -> None:
        self.name = name
        self.calls = 0

    async def judge(self, condition, frame, **kw) -> Verdict:  # pragma: no cover - not driven here
        self.calls += 1
        return Verdict(True, 0.9, "spy")

    async def detect(self, frame, goals) -> list:
        return []

    async def detect_events(self, frames, goals) -> list:
        return []


@dataclasses.dataclass(frozen=True)
class FakeBoundJudge:
    """Local stand-in for the pinned ``judge_registry.BoundJudge`` shape."""

    spec: ResolvedJudgeSpec
    adapter: Any
    static_context: Any = None


@contextlib.contextmanager
def installed_registry(get_or_create_adapter: Callable[[Any, Any], Any]):
    """Install a fake ``percept.judge_registry`` module (the one ``watchplan_apply`` lazily imports)
    for the duration of the block, then restore whatever was there before."""
    mod = types.ModuleType("percept.judge_registry")
    mod.get_or_create_adapter = get_or_create_adapter  # type: ignore[attr-defined]
    mod.BoundJudge = FakeBoundJudge  # type: ignore[attr-defined]
    import percept

    prev_sys = sys.modules.get("percept.judge_registry")
    prev_attr = getattr(percept, "judge_registry", None)
    sys.modules["percept.judge_registry"] = mod
    percept.judge_registry = mod  # type: ignore[attr-defined]
    try:
        yield mod
    finally:
        if prev_sys is not None:
            sys.modules["percept.judge_registry"] = prev_sys
        else:
            sys.modules.pop("percept.judge_registry", None)
        if prev_attr is not None:
            percept.judge_registry = prev_attr  # type: ignore[attr-defined]
        else:
            with contextlib.suppress(AttributeError):
                del percept.judge_registry


class JudgeInjectingResolver:
    """Wraps a real ``ConfigResolver`` and injects a ``ResolvedJudgeSpec`` onto the resolved plan,
    keyed by ``WatchPlan.id`` — so the (base-branch) resolver, which never sets ``.judge``, can still
    exercise the judge prepare path in these tests."""

    def __init__(self, judge_for: dict[str, ResolvedJudgeSpec]) -> None:
        self._inner = ConfigResolver()
        self._judge_for = judge_for

    def resolve(self, plan: WatchPlan, **kw):
        resolved = self._inner.resolve(plan, **kw)
        spec = self._judge_for.get(plan.id)
        if spec is not None:
            resolved = dataclasses.replace(resolved, judge=spec)
        return resolved

    @property
    def last_receipt(self):
        return self._inner.last_receipt


def _spec(ref: str, *, context_ref: str | None = None, context_sha256: str | None = None) -> ResolvedJudgeSpec:
    return ResolvedJudgeSpec(
        requested_ref=ref, resolved_ref=ref, factory_ref="factory@1", provider_ref="prov",
        model_requested="m", model_resolved="m", prompt_envelope_sha256=None,
        context_ref=context_ref, context_sha256=context_sha256,
    )


def _core() -> Percept:
    return Percept.create(vision=SpyJudge("vision"), discover_plugins=False)


def _plan(gid: str) -> WatchPlan:
    return WatchPlan(condition=f"a condition for {gid}", id=gid)


class AtomicApplyTests(unittest.TestCase):
    def test_success_path_arms_all_goals_and_registers_bindings(self):
        core = _core()
        resolver = JudgeInjectingResolver({"p1": _spec("a@1"), "p2": _spec("b@1")})
        adapters = {"a@1": SpyJudge("a"), "b@1": SpyJudge("b")}

        def factory(spec, cfg):
            return adapters[spec.resolved_ref]

        with installed_registry(factory):
            receipt = apply_watch_plan(core, [_plan("p1"), _plan("p2")], resolver=resolver)

        self.assertIn(receipt.status, ("applied", "applied_with_warnings"))
        self.assertEqual(set(receipt.watch_ids), {"p1", "p2"})
        self.assertIsNotNone(core.goals.get("p1"))
        self.assertIsNotNone(core.goals.get("p2"))
        # Bindings registered + goals marked judge-requesting, routed to the right adapter.
        self.assertIs(core._judge_bindings["p1"].adapter, adapters["a@1"])
        self.assertIs(core._judge_bindings["p2"].adapter, adapters["b@1"])
        self.assertEqual(core._judge_requesting_goal_ids, {"p1", "p2"})

    def test_factory_raise_on_second_goal_arms_nothing_and_leaves_prior_untouched(self):
        core = _core()
        # A pre-existing goal armed by an earlier (judge-less) apply — must stay untouched.
        apply_watch_plan(core, _plan("prior"))
        self.assertIsNotNone(core.goals.get("prior"))

        resolver = JudgeInjectingResolver({"p1": _spec("ok@1"), "p2": _spec("boom@1")})

        def factory(spec, cfg):
            if spec.resolved_ref == "boom@1":
                raise RuntimeError("factory exploded")
            return SpyJudge("ok")

        with installed_registry(factory):
            receipt = apply_watch_plan(core, [_plan("p1"), _plan("p2")], resolver=resolver)

        self.assertEqual(receipt.status, "rejected")
        self.assertTrue(receipt.dropped_fields)
        self.assertEqual(receipt.dropped_fields[0].code, "judge_adapter_unavailable")
        # ZERO goals from the batch armed; NO bindings registered; the prior goal is untouched.
        self.assertIsNone(core.goals.get("p1"))
        self.assertIsNone(core.goals.get("p2"))
        self.assertNotIn("p1", core._judge_bindings)
        self.assertNotIn("p2", core._judge_bindings)
        self.assertEqual(core._judge_requesting_goal_ids, set())
        self.assertIsNotNone(core.goals.get("prior"))

    def test_bad_context_hash_on_second_goal_arms_nothing(self):
        core = _core()
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "rubric.txt"
            artifact.write_text("REAL RUBRIC BYTES", encoding="utf-8")
            real_sha = hashlib.sha256(b"REAL RUBRIC BYTES").hexdigest()
            wrong_sha = "0" * 64

            resolver = JudgeInjectingResolver({
                "p1": _spec("ok@1"),
                "p2": _spec("ctx@1", context_ref="rubric.txt", context_sha256=wrong_sha),
            })
            deployment = DeploymentJudges(available_judges={}, context_store_root=tmp)

            with installed_registry(lambda spec, cfg: SpyJudge("ok")):
                receipt = apply_watch_plan(
                    core, [_plan("p1"), _plan("p2")], resolver=resolver, deployment_judges=deployment,
                )

            self.assertEqual(receipt.status, "rejected")
            self.assertEqual(receipt.dropped_fields[0].code, "judge_context_unverified")
            self.assertIsNone(core.goals.get("p1"))
            self.assertIsNone(core.goals.get("p2"))
            self.assertEqual(core._judge_bindings, {})

            # Sanity: with the CORRECT sha the same plan prepares + arms + verifies the exact bytes.
            resolver_ok = JudgeInjectingResolver({
                "p2": _spec("ctx@1", context_ref="rubric.txt", context_sha256=real_sha),
            })
            with installed_registry(lambda spec, cfg: SpyJudge("ok")):
                good = apply_watch_plan(core, _plan("p2"), resolver=resolver_ok, deployment_judges=deployment)
            self.assertIn(good.status, ("applied", "applied_with_warnings"))
            self.assertIsNotNone(core.goals.get("p2"))
            self.assertEqual(
                core._judge_bindings["p2"].static_context.decoded_text, "REAL RUBRIC BYTES"
            )

    def test_goal_removal_releases_binding(self):
        core = _core()
        resolver = JudgeInjectingResolver({"p1": _spec("a@1")})
        with installed_registry(lambda spec, cfg: SpyJudge("a")):
            apply_watch_plan(core, _plan("p1"), resolver=resolver)
        self.assertIn("p1", core._judge_bindings)
        self.assertIn("p1", core._judge_requesting_goal_ids)

        core.apply_goal_ops([GoalOp(op="cancel", goal_id="p1")])
        self.assertNotIn("p1", core._judge_bindings)
        self.assertNotIn("p1", core._judge_requesting_goal_ids)

    def test_cache_invalidation_failure_arms_nothing(self):
        core = _core()
        apply_watch_plan(core, _plan("prior"))
        goals_before = core.goals.snapshot()
        bindings_before = dict(core._watch_runtime_bindings)
        keys_before = dict(core._watch_runtime_keys)
        ledger_before = dict(core._watchplan_applied_hashes)

        def fail_cache(_goal_id):
            raise RuntimeError("cache backend unavailable")

        core.invalidate_cache = fail_cache
        receipt = apply_watch_plan(core, [_plan("p1"), _plan("p2")])

        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(receipt.dropped_fields[0].code, "commit_preflight_failed")
        self.assertEqual(core.goals.snapshot(), goals_before)
        self.assertEqual(core._watch_runtime_bindings, bindings_before)
        self.assertEqual(core._watch_runtime_keys, keys_before)
        self.assertEqual(core._watchplan_applied_hashes, ledger_before)

    def test_unexpected_commit_failure_restores_owned_runtime_state(self):
        core = _core()
        original_register = core._register_watch_runtime
        calls = 0

        def fail_second(goal_id, compiled, source_profile=None, application_key=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("binding store failed")
            return original_register(
                goal_id,
                compiled,
                source_profile,
                application_key=application_key,
            )

        core._register_watch_runtime = fail_second
        receipt = apply_watch_plan(core, [_plan("p1"), _plan("p2")])

        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(receipt.dropped_fields[0].code, "commit_failed_rolled_back")
        self.assertIsNone(core.goals.get("p1"))
        self.assertIsNone(core.goals.get("p2"))
        self.assertEqual(core._watch_runtime_bindings, {})
        self.assertEqual(core._watch_runtime_keys, {})
        self.assertEqual(core._judge_bindings, {})
        self.assertFalse(hasattr(core, "_watchplan_applied_hashes"))

    def test_distinct_plans_with_one_goal_id_reject_the_entire_batch(self):
        core = _core()
        receipt = apply_watch_plan(
            core,
            [
                WatchPlan(id="same", condition="first condition"),
                WatchPlan(id="same", condition="different condition"),
            ],
        )
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(receipt.dropped_fields[0].code, "duplicate_goal_id_in_batch")
        self.assertEqual(core.goals.all(), [])

    def test_goal_update_rejects_without_stranding_the_compiled_watch(self):
        core = _core()
        plan = _plan("p1")
        self.assertEqual(apply_watch_plan(core, plan).status, "applied")
        binding = core._watch_runtime_bindings["p1"]
        key = core._watch_runtime_keys["p1"]
        ledger = dict(core._watchplan_applied_hashes)

        with self.assertRaisesRegex(ValueError, "goal updates are unsupported"):
            core.apply_goal_ops([
                GoalOp(op="update", goal_id="p1", fields={"say": "updated notification"})
            ])

        self.assertIs(core._watch_runtime_bindings["p1"], binding)
        self.assertEqual(core._watch_runtime_keys["p1"], key)
        self.assertEqual(core._watchplan_applied_hashes, ledger)
        self.assertEqual(core.goals.get("p1").say, plan.condition)
        self.assertEqual(apply_watch_plan(core, plan).status, "applied")

    def test_restore_clears_session_local_compiled_and_apply_state(self):
        core = _core()
        self.assertEqual(apply_watch_plan(core, _plan("p1")).status, "applied")
        snap = core.snapshot()
        self.assertIn("p1", core._watch_runtime_bindings)
        self.assertTrue(core._watchplan_applied_hashes)

        core.restore(snap)

        self.assertIsNotNone(core.goals.get("p1"))
        self.assertEqual(core._watch_runtime_bindings, {})
        self.assertEqual(core._watch_runtime_keys, {})
        self.assertEqual(core._judge_bindings, {})
        self.assertEqual(core._judge_requesting_goal_ids, set())
        self.assertFalse(core._watchplan_applied_hashes)

    def test_restore_can_rebind_the_exact_executable_without_replacing_restored_cognition(self):
        core = _core()
        profile = SourceProfile(
            name="restore-source",
            modalities=frozenset({"video"}),
            temporal_unit="frame",
            admission_strategy="pixel-change",
            evidence_strategy="frame",
            identity_strategy="track_continuity",
        )
        plan = WatchPlan(
            id="p1",
            condition="a condition for p1",
            consequence={"miss_cost": "severe"},
            rearm="deadline",
            explicit={
                "Goal.state.deadline_after_s": 1.0,
            },
        )
        resolver = JudgeInjectingResolver({"p1": _spec("judge@1")})
        adapters = []

        def factory(spec, cfg):
            adapter = SpyJudge(f"adapter-{len(adapters)}")
            adapters.append(adapter)
            return adapter

        with installed_registry(factory):
            first = apply_watch_plan(
                core, plan, resolver=resolver, source_profile=profile
            )
        self.assertIn(first.status, ("applied", "applied_with_warnings"))
        core.goals.get("p1").state["count"] = 7
        core.gate.evaluate(("p1", None), Verdict(False, 0.99, "restored no"))
        snap = core.snapshot()

        core.restore(snap)
        restored_goal = core.goals.get("p1")
        restored_gate = core.gate.snapshot()
        self.assertEqual(restored_goal.state["count"], 7)
        self.assertEqual(core._watch_runtime_bindings, {})
        self.assertEqual(core._judge_bindings, {})
        self.assertEqual(core._watch_rebind_required, {"p1"})

        # Restored compiled goals are active-but-quarantined until exact rebind. They cannot fall
        # through to the core default judge, worker gate, or absence/boundary paths.
        default_calls = core.vision.calls
        self.assertEqual(
            asyncio.run(core.perceive_judged("frame", "camera", t=100.0)),
            [],
        )
        self.assertEqual(core.vision.calls, default_calls)
        core.last_supported["p1"] = 0.0
        overdue_t = restored_goal.state["deadline_at"] + 100.0
        restored_goal.state["type"] = "process"
        ctx = core.executor.ctx("p1")
        ctx.process_log = [{"step": "restored"}]
        ctx.last_step_at = 0.0
        self.assertEqual(core.check_observability(t=overdue_t, staleness_s=1.0), [])
        self.assertEqual(core.check_deadlines(t=overdue_t), [])
        self.assertEqual(
            asyncio.run(core.check_episode_boundary(t=overdue_t, inactivity_s=1.0)),
            [],
        )
        restored_goal.state["type"] = "watch"

        async def collect_restored():
            stats = RuntimeStats()
            frames = [
                (0.0, gray_frame(8, 8, 0), "prime"),
                (1.0, gray_frame(8, 8, 255), "changed"),
            ]
            incidents = [
                incident
                async for incident in run(
                    core, source=frames, stats=stats, judge_dispatch="inline"
                )
            ]
            return incidents, stats

        incidents, stats = asyncio.run(collect_restored())
        self.assertEqual(incidents, [])
        self.assertEqual(stats.escalations, 0)
        self.assertEqual(core.vision.calls, default_calls)

        with installed_registry(factory):
            rebound = apply_watch_plan(
                core, plan, resolver=resolver, source_profile=profile
            )

        self.assertIn(rebound.status, ("applied", "applied_with_warnings"))
        self.assertIs(core.goals.get("p1"), restored_goal)
        self.assertEqual(restored_goal.state["count"], 7)
        self.assertEqual(core.gate.snapshot(), restored_gate)
        self.assertIs(core._watch_runtime_bindings["p1"][1], profile)
        self.assertIs(core._judge_bindings["p1"].adapter, adapters[-1])
        self.assertEqual(core._watch_rebind_required, set())
        self.assertEqual(
            core._watch_runtime_keys["p1"],
            core._watch_executable_identities["p1"],
        )
        rebound_adapter = core._judge_bindings["p1"].adapter
        asyncio.run(core.perceive_judged("frame", "camera", t=101.0))
        self.assertEqual(rebound_adapter.calls, 1)

    def test_cancelled_goal_id_is_not_reusable_even_for_the_same_watch(self):
        core = _core()
        plan = _plan("p1")
        self.assertEqual(apply_watch_plan(core, plan).status, "applied")
        core.apply_goal_ops([GoalOp(op="cancel", goal_id="p1")])

        receipt = apply_watch_plan(core, plan)

        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(
            receipt.dropped_fields[0].code,
            "active_goal_id_reconfiguration_unsupported",
        )
        self.assertIn(
            "completion or cancellation does not make the old id reusable",
            receipt.dropped_fields[0].message,
        )

    def test_mixed_goal_op_batch_is_prevalidated_before_an_earlier_update_mutates(self):
        core = _core()
        core.add_goal(Goal(id="p1", condition="original", say="original"))

        with self.assertRaisesRegex(KeyError, "missing"):
            core.apply_goal_ops([
                GoalOp(op="update", goal_id="p1", fields={"say": "mutated"}),
                GoalOp(op="complete", goal_id="missing"),
            ])

        self.assertEqual(core.goals.get("p1").say, "original")
        self.assertEqual(core.goals.status("p1"), "active")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
