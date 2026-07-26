"""— the PUBLIC recipe: a claim_slot plan + live model-reply JSON forms a revision chain.

This is the end-to-end proof that closes P05-SDK-004. Unlike ``test_y20_live_supersession_delivery.py``
(which hands the fold PRE-BUILT ``Verdict(operation="replace", ...)`` objects), here the scripted vision
returns **model-reply-shaped JSON TEXT** and the REAL backend parser
(:func:`percept.backends.anthropic._parse_verdict`, claim_mode) is what turns it into a mutating verdict.
So the whole chain is exercised: ``apply_watch_plan`` arms a ``claim_slot`` goal → ``run()`` (worker
delivery path) → ``agent.perceive_judged`` marks ``context["claim_mode"]`` → ``_judge`` forwards it →
the parser folds ``operation``/``value`` → the authority-gated claim fold mutates → a superseding
revision reaches a sink double.

Three scenarios:
  A. assert "$40K", then a CORRECTION reply (operation="replace", value="$25K") → the delivered
     incidents carry the supersession chain (revision_id + supersedes).
  B. assert "$40K", then a QUESTION reply (operation stays "assert") → NO mutation, no chain; the
     slot value is unchanged.
  C. assert "$40K", then a RETRACTION reply (operation="retract") → a superseding revision with the
     value cleared.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Percept
from percept.backends.anthropic import _parse_verdict
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.contracts.watchplan import WatchPlan
from percept.harness.tier0 import gray_frame
from percept.runtime import incident_payload, run
from percept.watchplan_apply import apply_watch_plan

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)


class _ReplyShapedJudge(FakeVision):
    """A scripted vision double that returns MODEL-REPLY JSON text run through the REAL claim parser.

    Keyed off the judge-frame token (like :class:`FakeVision`), but instead of a pre-built ``Verdict``
    each token maps to the raw JSON STRING a real model would emit; :func:`_parse_verdict` (claim_mode
    threaded off ``context["claim_mode"]``) is what produces the verdict. Subclasses ``FakeVision`` so it
    inherits ``PARSES_CLAIM_CHANNEL`` (no receipt warning) and ``detect``/``detect_events``."""

    def __init__(self, replies: dict[str, str]) -> None:
        super().__init__()
        self._replies = dict(replies)

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None):
        text = self._replies[str(frame)]
        claim_mode = bool(context and context.get("claim_mode"))
        return _parse_verdict(text, claim_mode=claim_mode)


def _run(core, source):
    delivered: list = []

    async def _go():
        incs = [inc async for inc in run(core, source=source, judge_dispatch="inline",
                                          on_incident=delivered.append)]
        return incs

    incidents = asyncio.run(_go())
    return incidents, delivered


def _core(replies: dict[str, str]):
    settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=10_000.0))
    core = Percept.create(vision=_ReplyShapedJudge(replies), settings=settings, discover_plugins=False)
    plan = WatchPlan(
        condition="a buyer states the approved budget",
        explicit={"Goal.state.identity_policy": "claim_slot"},
    )
    receipt = apply_watch_plan(core, plan)
    assert receipt.watch_ids, receipt
    # a parsing-capable backend must NOT earn the capability warning
    assert "claim_mode_backend_unverified" not in [w.code for w in receipt.warnings], receipt
    return core, receipt.watch_ids[0]


# The STILL->MOVED motion edges are the rising edges the gate escalates on: one
# for the original claim, one (after a re-arm back to STILL) for the later correction/question/retract.
def _source(first_token: str, second_token: str):
    return [
        (0.0, STILL, first_token),    # prime: no motion yet
        (1.0, MOVED, first_token),    # still->moved: escalate -> assert -> FIRE (provisional)
        (2.0, STILL, first_token),    # moved->still: re-arm the motion edge
        (25.0, MOVED, second_token),  # still->moved: escalate -> the second reply
    ]


_ASSERT_40K = '{"satisfied": true, "confidence": 0.9, "evidence": "the budget is $40K", "operation": "assert", "value": "$40K"}'
_REPLACE_25K = '{"satisfied": true, "confidence": 0.9, "evidence": "actually, the approved budget is $25K", "operation": "replace", "value": "$25K"}'
# A QUESTION — the model correctly does NOT treat it as a correction (operation stays "assert").
_QUESTION_25K = '{"satisfied": true, "confidence": 0.9, "evidence": "buyer asks: could we do $25K?", "operation": "assert", "value": "$25K"}'
_RETRACT = '{"satisfied": true, "confidence": 0.8, "evidence": "never mind, forget the budget", "operation": "retract"}'


class TestScenarioA_CorrectionFormsChain(unittest.TestCase):
    def test_replace_reply_supersedes_the_original(self):
        core, gid = _core({"assert_40k": _ASSERT_40K, "replace_25k": _REPLACE_25K})
        incidents, delivered = _run(core, _source("assert_40k", "replace_25k"))

        self.assertEqual(len(incidents), 2, "the original claim and the correction both fire")
        self.assertEqual([i.id for i in delivered], [i.id for i in incidents])
        original, correction = incidents

        # ORIGINAL: its own revision id, no supersedes link.
        self.assertIsNotNone(original.revision_id)
        self.assertIsNone(original.supersedes)
        # CORRECTION: a different revision id that supersedes the original's — on the object AND the wire.
        self.assertIsNotNone(correction.revision_id)
        self.assertNotEqual(correction.revision_id, original.revision_id)
        self.assertEqual(correction.supersedes, original.revision_id)
        self.assertEqual(incident_payload(correction)["supersedes"], original.revision_id)

        # The slot's authoritative value is the corrected one.
        self.assertEqual(core.episode_current(gid).value, "$25K")


class TestScenarioB_QuestionIsNotACorrection(unittest.TestCase):
    def test_question_reply_never_mutates_or_chains(self):
        core, gid = _core({"assert_40k": _ASSERT_40K, "question_25k": _QUESTION_25K})
        incidents, delivered = _run(core, _source("assert_40k", "question_25k"))

        # A question is not a correction: no delivered incident carries a supersession link ...
        self.assertTrue(incidents, "the original assert still fires")
        self.assertTrue(all(i.supersedes is None for i in incidents))
        self.assertTrue(all(incident_payload(i)["supersedes"] is None for i in incidents))
        # ... and the slot value is unchanged — the $25K proposal never overwrote the $40K claim.
        self.assertEqual(core.episode_current(gid).value, "$40K")


class TestScenarioC_RetractionClearsValue(unittest.TestCase):
    def test_retract_reply_supersedes_and_clears(self):
        core, gid = _core({"assert_40k": _ASSERT_40K, "retract": _RETRACT})
        incidents, delivered = _run(core, _source("assert_40k", "retract"))

        self.assertEqual(len(incidents), 2, "the original claim and the retraction both fire")
        original, retraction = incidents
        # The retraction supersedes the original revision ...
        self.assertIsNotNone(original.revision_id)
        self.assertEqual(retraction.supersedes, original.revision_id)
        self.assertEqual(incident_payload(retraction)["supersedes"], original.revision_id)
        # ... and clears the slot value (a withdrawal ends the revision line).
        self.assertIsNone(core.episode_current(gid).value)


if __name__ == "__main__":
    unittest.main()
