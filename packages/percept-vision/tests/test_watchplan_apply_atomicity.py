"""P06-001 two-phase atomic ``apply_watch_plan`` — a batch arms every goal or none.

Phase 1 PREPARE builds every goal + every judge binding with NO runtime mutation; Phase 2 COMMIT
arms the goals and registers the bindings only after the whole batch prepared. A judge PREPARE
failure (a factory that raises, or a static context that fails its digest check) must leave the
runtime untouched and surface a structured rejected receipt.

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

from percept.adapters import Verdict
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

    async def judge(self, condition, frame, **kw) -> Verdict:  # pragma: no cover - not driven here
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
