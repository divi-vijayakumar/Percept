"""P06-001 full-stack acceptance suite — the incident's eight criteria asserted END-TO-END on the
REAL stack: the real :class:`percept.resolver.ConfigResolver`, the real
:mod:`percept.judge_registry` authorization + adapter cache, the real
:class:`percept.judge_context.JudgeContextStore`, the real two-phase
:func:`percept.watchplan_apply.apply_watch_plan`, and a real :class:`percept.agent.Percept` driven
in the offline harness — with a fake SPECIALIST vision factory installed through the genuine
``percept.registry.register`` seam (never a monkeypatched resolver or an injected judge spec).

Pieces of these are pinned piecemeal across the wave test files (``test_judge_selection`` for the
resolver, ``test_judge_dispatch`` for the dispatch site on local doubles, ``test_watchplan_apply_atomicity``
for atomic apply on a fake registry). This suite is the one place they are exercised together on the
real objects, so a regression in ANY seam that only shows up at the seams is caught here.

The eight criteria (incident PERCEPT-SDK-P06-001):
  1. Known judge resolution moves ``component_snapshot_hash``; receipt ``judge_requested == judge_resolved``.
  2. Unknown judge fails pre-arm (typed refusal); zero adapter/model calls; zero goals armed.
  3. A known context loads by exact ref+sha; the sha rides the receipt AND the judge.started/finished trace.
  4. A corrupted context hash is a hard pre-judgment failure: zero model calls, zero generalist fallback,
     nothing armed (atomicity).
  5. The registered specialist adapter receives the judgment; the generalist ``self.vision`` gets ZERO
     judge calls for that goal.
  6. Neutral vs specialist resolve of the same plan differ ONLY in the judge block (shared core equal
     under ``hash_projection`` excluding the judge pointers); plan_hash and component_snapshot_hash differ.
  7. An incident emitted through the specialist path keeps the Contract 1.9 evidence shape and
     ``verify_evidence_links`` resolves it.
  8. The receipt carries judge_requested/resolved, judge_context_ref/sha256, and fallback_used=False.
"""
from __future__ import annotations

import asyncio
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from percept import registry
from percept.adapters import Verdict
from percept.agent import Percept
from percept.config import GateSettings, Settings
from percept.contracts import ArtifactBinding
from percept.contracts.watchplan import WatchPlan
from percept.deployment_judges import DeploymentJudges, JudgeDeploymentSpec
from percept.gate import FIRE, Gate
from percept.incidents import make_incident
from percept.judge_context import JUDGE_STATIC_CONTEXT_KEY, JudgeContextError
from percept.judge_registry import clear_adapter_cache
from percept.replay import gate_config_of, verify_evidence_links
from percept.resolver import ConfigResolver, hash_projection
from percept.tracing import (
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGE_FINISHED,
    STAGE_JUDGE_STARTED,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
)
from percept.watchplan_apply import apply_watch_plan

#: The one specialist judge the deployment authorizes, and the vision-seam factory that implements it.
JUDGE_REF = "specialist@1"
FACTORY_REF = "acceptance-specialist-factory@1"
RUBRIC_TEXT = "RUBRIC: fire only when the specialist condition holds"


class _RecordingJudge:
    """A vision-seam double that records every ``judge`` call (condition + received context) and returns
    a scripted satisfied+confident verdict. Serves as either the generalist ``self.vision`` or the
    registered specialist adapter. ``detect`` returns nothing so it drives the offline watch loop."""

    def __init__(self, name: str, *, model: str | None = None) -> None:
        self.name = name
        self.model = model
        self.judge_calls: list[dict[str, Any]] = []

    async def judge(self, condition: str, frame: Any, *, roi: Any = None, exemplar: Any = None,
                    corrections: Any = None, frames: list | None = None, audio: Any = None,
                    context: dict | None = None) -> Verdict:
        self.judge_calls.append({"condition": condition, "context": context})
        return Verdict(satisfied=True, confidence=0.95, evidence=f"{self.name}:{condition}")

    async def detect(self, frame: Any, goals: list) -> list:
        return []

    async def detect_events(self, frames: list, goals: list) -> list:
        return []


def _settings() -> Settings:
    return Settings(gate=GateSettings(threshold=0.7, validity_window_s=10.0))


def _recorder() -> TraceRecorder:
    manifest = RunTraceManifest.create(
        run_id="run-p06-acceptance", source="camera", policy=TracePolicy(level="full"),
        gate_config=gate_config_of(Gate()),
    )
    return TraceRecorder(manifest)


class P06AcceptanceSuite(unittest.TestCase):
    """Each test builds the whole real stack; a per-test context-store dir + a fresh adapter cache keep
    the process-level judge adapter memo from leaking a spy between cases."""

    def setUp(self) -> None:
        clear_adapter_cache()
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(clear_adapter_cache)
        self.ctx_root = self._tmp.name
        artifact = Path(self.ctx_root) / "rubric.txt"
        artifact.write_bytes(RUBRIC_TEXT.encode("utf-8"))
        self.rubric_sha = hashlib.sha256(RUBRIC_TEXT.encode("utf-8")).hexdigest()
        # A FRESH specialist adapter per test, installed through the real registry seam. The factory
        # records how many times it was constructed so "zero adapter calls" is provable, not assumed.
        self.specialist = _RecordingJudge("specialist", model="specialist-model@1")
        self.factory_builds = 0

        def _factory(cfg: Any) -> _RecordingJudge:
            self.factory_builds += 1
            return self.specialist

        registry.register("vision", FACTORY_REF, _factory)
        self.generalist = _RecordingJudge("generalist", model="generalist-model")

    def _deployment(self, *, supports_static_context: bool = True) -> DeploymentJudges:
        return DeploymentJudges(
            available_judges={
                JUDGE_REF: JudgeDeploymentSpec(
                    factory_ref=FACTORY_REF, provider_ref="prov", model_ref="specialist-model@1",
                    supports_static_context=supports_static_context,
                )
            },
            context_store_root=self.ctx_root,
        )

    def _core(self) -> Percept:
        return Percept.create(vision=self.generalist, settings=_settings(), discover_plugins=False)

    def _specialist_plan(self, gid: str = "sg", *, condition: str = "the door is open",
                         with_context: bool = True) -> WatchPlan:
        binding = ArtifactBinding(ref="rubric.txt", sha256=self.rubric_sha) if with_context else None
        return WatchPlan(condition=condition, id=gid, judge=JUDGE_REF, judge_context=binding)

    def _neutral_plan(self, gid: str = "sg", *, condition: str = "the door is open") -> WatchPlan:
        return WatchPlan(condition=condition, id=gid)

    # -- Criterion 1 -----------------------------------------------------------------------------
    def test_c1_known_judge_moves_component_hash_and_receipt_agrees(self):
        resolver = ConfigResolver()
        deployment = self._deployment()
        specialist = resolver.resolve(self._specialist_plan(), deployment_judges=deployment)
        receipt = resolver.last_receipt
        neutral = resolver.resolve(self._neutral_plan(), deployment_judges=deployment)

        # The requested specialist judge changes the component snapshot hash.
        self.assertNotEqual(specialist.component_snapshot_hash, neutral.component_snapshot_hash)
        self.assertIn("judge", specialist.components)
        # The resolve receipt agrees on requested vs resolved (a known judge resolves to itself).
        self.assertEqual(receipt.judge_requested, JUDGE_REF)
        self.assertEqual(receipt.judge_resolved, JUDGE_REF)
        self.assertEqual(receipt.judge_requested, receipt.judge_resolved)

    # -- Criterion 2 -----------------------------------------------------------------------------
    def test_c2_unknown_judge_fails_pre_arm_with_zero_adapter_and_model_calls(self):
        core = self._core()
        deployment = self._deployment()
        plan = WatchPlan(condition="x", id="ghost", judge="not-in-catalog@9")

        receipt = apply_watch_plan(
            core, plan, resolver=ConfigResolver(), deployment_judges=deployment,
        )

        self.assertEqual(receipt.status, "rejected")
        self.assertTrue(receipt.dropped_fields)
        # Fail-closed PRE-ARM: the factory was never constructed, no goal armed, no binding registered.
        self.assertEqual(self.factory_builds, 0)
        self.assertEqual(self.specialist.judge_calls, [])
        self.assertEqual(self.generalist.judge_calls, [])
        self.assertIsNone(core.goals.get("ghost"))
        self.assertEqual(core._judge_bindings, {})
        self.assertEqual(core._judge_requesting_goal_ids, set())

    # -- Criterion 3 -----------------------------------------------------------------------------
    def test_c3_known_context_loads_and_sha_rides_receipt_and_trace(self):
        core = self._core()
        deployment = self._deployment()
        resolver = ConfigResolver()
        receipt = apply_watch_plan(
            core, self._specialist_plan(gid="sg"), resolver=resolver, deployment_judges=deployment,
        )
        self.assertIn(receipt.status, ("applied", "applied_with_warnings"))
        # The store loaded the EXACT bytes for ref rubric.txt at the pinned sha.
        self.assertEqual(core._judge_bindings["sg"].static_context.decoded_text, RUBRIC_TEXT)
        self.assertEqual(core._judge_bindings["sg"].static_context.sha256, self.rubric_sha)
        # The sha rides the ADR-009 resolve receipt.
        self.assertEqual(resolver.last_receipt.judge_context_sha256, self.rubric_sha)

        events: list = []
        core.on_trace = events.append
        asyncio.run(core.perceive_judged("frame", t=1.0))

        for stage in (STAGE_JUDGE_STARTED, STAGE_JUDGE_FINISHED):
            rows = [e.data for e in events if e.stage == stage and e.goal_id == "sg"]
            self.assertEqual(len(rows), 1, f"expected one {stage} for sg")
            self.assertEqual(rows[0]["judge_context_sha256"], self.rubric_sha)
            self.assertEqual(rows[0]["judge_resolved"], JUDGE_REF)
            self.assertIs(rows[0]["fallback_used"], False)

    # -- Criterion 4 -----------------------------------------------------------------------------
    def test_c4_corrupted_context_hash_is_a_hard_pre_arm_failure_arming_nothing(self):
        core = self._core()
        deployment = self._deployment()
        bad = WatchPlan(
            condition="the door is open", id="bad", judge=JUDGE_REF,
            judge_context=ArtifactBinding(ref="rubric.txt", sha256="0" * 64),
        )
        # The bytes on disk do not match the pinned digest -> a hard pre-judgment failure at resolve
        # (the store verifies the exact bytes), raised before any adapter/model call. The two-phase
        # apply arms nothing: prior + this goal absent, no binding, and no judge call on EITHER seam.
        with self.assertRaises(JudgeContextError):
            apply_watch_plan(core, bad, resolver=ConfigResolver(), deployment_judges=deployment)

        self.assertIsNone(core.goals.get("bad"))
        self.assertEqual(core._judge_bindings, {})
        self.assertEqual(core._judge_requesting_goal_ids, set())
        self.assertEqual(self.specialist.judge_calls, [])  # zero model calls
        self.assertEqual(self.generalist.judge_calls, [])  # zero generalist fallback

    # -- Criterion 5 -----------------------------------------------------------------------------
    def test_c5_route_spy_specialist_judges_generalist_gets_zero_calls_for_the_goal(self):
        core = self._core()
        deployment = self._deployment()
        apply_watch_plan(
            core, self._specialist_plan(gid="sg"), resolver=ConfigResolver(),
            deployment_judges=deployment,
        )
        asyncio.run(core.perceive_judged("frame", t=1.0))

        # The specialist adapter received the judgment; the generalist self.vision judged ZERO times.
        self.assertEqual([c["condition"] for c in self.specialist.judge_calls], ["the door is open"])
        self.assertEqual(self.generalist.judge_calls, [])
        # And the injected static context reached the specialist under the reserved key.
        self.assertEqual(
            self.specialist.judge_calls[0]["context"][JUDGE_STATIC_CONTEXT_KEY], RUBRIC_TEXT
        )

    # -- Criterion 6 -----------------------------------------------------------------------------
    def test_c6_neutral_vs_specialist_differ_only_in_the_judge_block(self):
        resolver = ConfigResolver()
        deployment = self._deployment()
        specialist = resolver.resolve(self._specialist_plan(gid="sg"), deployment_judges=deployment)
        neutral = resolver.resolve(self._neutral_plan(gid="sg"), deployment_judges=deployment)

        # SHARED CORE equality: excluding the judge pointers, the two component snapshots are identical.
        shared_specialist = hash_projection(dict(specialist.components), ["/judge"])
        shared_neutral = hash_projection(dict(neutral.components), ["/judge"])
        self.assertEqual(shared_specialist, shared_neutral)
        # Only the specialist snapshot carries the judge block.
        self.assertIn("judge", specialist.components)
        self.assertNotIn("judge", neutral.components)
        # The plan hash differs (judge + judge_context wire fields) and so does the component snapshot.
        self.assertNotEqual(specialist.plan_hash, neutral.plan_hash)
        self.assertNotEqual(specialist.component_snapshot_hash, neutral.component_snapshot_hash)

    # -- Criterion 7 -----------------------------------------------------------------------------
    def test_c7_specialist_incident_keeps_contract_1_9_shape_and_links_resolve(self):
        core = self._core()
        deployment = self._deployment()
        rec = _recorder()
        core.on_trace = rec.tracer()
        apply_watch_plan(
            core, self._specialist_plan(gid="sg"), resolver=ConfigResolver(),
            deployment_judges=deployment,
        )

        async def run():
            fire = None
            for e in await core.perceive_judged("frame", t=1.0):
                if e.action == FIRE:
                    fire = e
            return fire

        fire = asyncio.run(run())
        self.assertIsNotNone(fire, "the specialist path must produce a fire")
        # Contract 1.9 evidence shape: the fire carries a real JudgmentEvent with a non-null evidence_seq.
        self.assertIsNotNone(fire.judgment)
        self.assertIsNotNone(fire.judgment.evidence_seq)

        inc = make_incident(fire, ts=99.0)
        # The delivered Incident retains the 1.9 evidence-link fields.
        self.assertIsNotNone(inc.judgment_event_id)
        self.assertIsNotNone(inc.evidence_seq)
        rec.record(
            STAGE_INCIDENT_CREATED, trace_id="camera@99.0", source_ts=99.0,
            goal_id=inc.concern_id, incident_id=inc.id, entity_id=inc.entity_id,
            judgment_event_id=inc.judgment_event_id, evidence_origin=inc.evidence_origin,
        )
        report = verify_evidence_links(rec.records)
        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.resolved, 1)
        self.assertEqual(report.violations, ())

    # -- Criterion 8 -----------------------------------------------------------------------------
    def test_c8_receipt_carries_all_judge_provenance_with_fallback_false(self):
        resolver = ConfigResolver()
        deployment = self._deployment()
        core = self._core()
        apply_watch_plan(
            core, self._specialist_plan(gid="sg"), resolver=resolver, deployment_judges=deployment,
        )
        receipt = resolver.last_receipt  # the ADR-009 resolve receipt carries the judge provenance
        self.assertEqual(receipt.judge_requested, JUDGE_REF)
        self.assertEqual(receipt.judge_resolved, JUDGE_REF)
        self.assertEqual(receipt.judge_context_ref, "rubric.txt")
        self.assertEqual(receipt.judge_context_sha256, self.rubric_sha)
        self.assertIs(receipt.fallback_used, False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
