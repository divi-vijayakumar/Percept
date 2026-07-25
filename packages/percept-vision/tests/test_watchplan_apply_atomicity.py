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
from percept.resolver import ConfigResolver
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

    def test_safe_say_update_preserves_runtime_but_invalidates_apply_receipt(self):
        core = _core()
        plan = _plan("p1")
        resolver = JudgeInjectingResolver({"p1": _spec("judge@1")})
        with installed_registry(lambda spec, cfg: SpyJudge()):
            self.assertIn(
                apply_watch_plan(core, plan, resolver=resolver).status,
                ("applied", "applied_with_warnings"),
            )
        binding = core._watch_runtime_bindings["p1"]
        judge_binding = core._judge_bindings["p1"]

        core.goals.apply([
            GoalOp(op="update", goal_id="p1", fields={"say": "updated notification"})
        ])

        self.assertIs(core._watch_runtime_bindings["p1"], binding)
        self.assertIs(core._judge_bindings["p1"], judge_binding)
        self.assertIn("p1", core._judge_requesting_goal_ids)
        self.assertNotIn("p1", core._watch_runtime_keys)
        self.assertEqual(core._watchplan_applied_hashes, {})
        self.assertEqual(core.goals.get("p1").say, "updated notification")
        # The old plan cannot false-hit its stale receipt after presentation changed.
        self.assertEqual(
            apply_watch_plan(core, plan, resolver=resolver).status,
            "rejected",
        )

    def test_direct_registry_complete_and_cancel_release_runtime_and_judge(self):
        for operation in ("complete", "cancel"):
            with self.subTest(operation=operation):
                core = _core()
                plan = _plan("p1")
                resolver = JudgeInjectingResolver({"p1": _spec("judge@1")})

                with installed_registry(lambda spec, cfg: SpyJudge()):
                    receipt = apply_watch_plan(core, plan, resolver=resolver)
                self.assertIn(receipt.status, ("applied", "applied_with_warnings"))
                self.assertIn("p1", core._watch_runtime_bindings)
                self.assertIn("p1", core._judge_bindings)

                core.goals.apply([GoalOp(op=operation, goal_id="p1")])

                expected_status = "cancelled" if operation == "cancel" else "complete"
                self.assertEqual(core.goals.status("p1"), expected_status)
                self.assertNotIn("p1", core._watch_runtime_bindings)
                self.assertNotIn("p1", core._watch_runtime_keys)
                self.assertNotIn("p1", core._judge_bindings)
                self.assertNotIn("p1", core._judge_requesting_goal_ids)
                self.assertEqual(core._watchplan_applied_hashes, {})

    def test_cached_receipt_requires_intact_live_executable_state(self):
        def mismatch_runtime(core):
            compiled, source_profile = core._watch_runtime_bindings["p1"]
            core._watch_runtime_bindings["p1"] = (
                dataclasses.replace(compiled, plan_mode="window"),
                source_profile,
            )

        def replace_goal(core, **changes):
            current = core.goals.get("p1")
            core.goals._goals["p1"] = dataclasses.replace(current, **changes)

        def mutate_state(core, key):
            core.goals.get("p1").state[key] = {"tampered": True}

        attacks = {
            "missing runtime binding": lambda core: core._watch_runtime_bindings.pop("p1"),
            "mismatched runtime executable": mismatch_runtime,
            "missing runtime key": lambda core: core._watch_runtime_keys.pop("p1"),
            "missing ledger": lambda core: core._watchplan_applied_hashes.clear(),
            "changed say": lambda core: replace_goal(core, say="tampered"),
            "changed condition": lambda core: replace_goal(core, condition="tampered"),
            "changed schedule": lambda core: mutate_state(core, "schedule"),
            "changed perception": lambda core: mutate_state(core, "perception"),
            "changed behavior": lambda core: mutate_state(core, "behavior"),
            "changed detectors": lambda core: mutate_state(core, "detectors"),
            "changed sense": lambda core: mutate_state(core, "sense"),
        }
        for name, attack in attacks.items():
            with self.subTest(name=name):
                core = _core()
                plan = _plan("p1")
                self.assertEqual(apply_watch_plan(core, plan).status, "applied")
                attack(core)

                self.assertEqual(apply_watch_plan(core, plan).status, "rejected")

    def test_cached_judged_receipt_requires_the_exact_usable_judge_binding(self):
        def missing(core):
            core._judge_bindings.pop("p1")

        def adapter_none(core):
            prior = core._judge_bindings["p1"]
            replacement = dataclasses.replace(prior, adapter=None)
            core._judge_bindings["p1"] = replacement
            core._judge_binding_tokens["p1"] = id(replacement)

        def replaced(core):
            prior = core._judge_bindings["p1"]
            core._judge_bindings["p1"] = dataclasses.replace(
                prior, adapter=SpyJudge("replacement")
            )

        for name, attack in {
            "missing": missing,
            "adapter none": adapter_none,
            "replaced binding": replaced,
        }.items():
            with self.subTest(name=name):
                core = _core()
                plan = _plan("p1")
                resolver = JudgeInjectingResolver({"p1": _spec("judge@1")})
                with installed_registry(lambda spec, cfg: SpyJudge()):
                    self.assertIn(
                        apply_watch_plan(core, plan, resolver=resolver).status,
                        ("applied", "applied_with_warnings"),
                    )
                attack(core)

                self.assertEqual(
                    apply_watch_plan(core, plan, resolver=resolver).status,
                    "rejected",
                )

    def test_cached_judge_context_rechecks_bound_bytes_against_digest(self):
        core = _core()
        plan = _plan("p1")
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "rubric.txt"
            artifact.write_bytes(b"VERIFIED RUBRIC")
            digest = hashlib.sha256(b"VERIFIED RUBRIC").hexdigest()
            resolver = JudgeInjectingResolver({
                "p1": _spec(
                    "judge@1",
                    context_ref="rubric.txt",
                    context_sha256=digest,
                )
            })
            deployment = DeploymentJudges(available_judges={}, context_store_root=tmp)
            with installed_registry(lambda spec, cfg: SpyJudge()):
                self.assertIn(
                    apply_watch_plan(
                        core,
                        plan,
                        resolver=resolver,
                        deployment_judges=deployment,
                    ).status,
                    ("applied", "applied_with_warnings"),
                )
            prior = core._judge_bindings["p1"]
            bad_context = dataclasses.replace(
                prior.static_context,
                content_bytes=b"TAMPERED RUBRIC",
            )
            replacement = dataclasses.replace(prior, static_context=bad_context)
            core._judge_bindings["p1"] = replacement
            core._judge_binding_tokens["p1"] = id(replacement)

            self.assertEqual(
                apply_watch_plan(
                    core,
                    plan,
                    resolver=resolver,
                    deployment_judges=deployment,
                ).status,
                "rejected",
            )

    def test_compiled_watch_snapshot_refuses_before_producing_an_artifact(self):
        core = _core()
        self.assertEqual(apply_watch_plan(core, _plan("p1")).status, "applied")
        import percept
        from percept import CompiledWatchSnapshotUnsupportedError

        self.assertIn("CompiledWatchSnapshotUnsupportedError", percept.__all__)

        class RecordingPersistence:
            def __init__(self):
                self.saved = []

            def save(self, key, value):
                self.saved.append((key, value))

        persistence = RecordingPersistence()
        with self.assertRaisesRegex(
            CompiledWatchSnapshotUnsupportedError, "do not support compiled Watches"
        ):
            core.save_state(persistence)

        self.assertEqual(persistence.saved, [])
        self.assertIn("p1", core._watch_runtime_bindings)
        self.assertIsNotNone(core.goals.get("p1"))

    def test_direct_registry_restore_releases_prior_compiled_owner_state(self):
        core = _core()
        plan = _plan("p1")
        resolver = JudgeInjectingResolver({"p1": _spec("judge@1")})
        with installed_registry(lambda spec, cfg: SpyJudge()):
            self.assertIn(
                apply_watch_plan(core, plan, resolver=resolver).status,
                ("applied", "applied_with_warnings"),
            )
        registry_snapshot = core.goals.snapshot()

        core.goals.restore(registry_snapshot)

        self.assertIsNotNone(core.goals.get("p1"))
        self.assertEqual(core._watch_runtime_bindings, {})
        self.assertEqual(core._watch_runtime_keys, {})
        self.assertEqual(core._judge_bindings, {})
        self.assertEqual(core._judge_binding_tokens, {})
        self.assertEqual(core._judge_requesting_goal_ids, set())
        self.assertEqual(core._watchplan_applied_hashes, {})

    def test_plain_goal_snapshot_restore_remains_supported(self):
        core = _core()
        core.add_goal(Goal(id="plain", condition="plain condition", say="plain"))
        snap = core.snapshot()
        self.assertNotIn("watch_executable_identities", snap)

        restored = _core().restore(snap)

        self.assertEqual(restored.goals.get("plain").condition, "plain condition")
        self.assertEqual(restored.goals.status("plain"), "active")

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
