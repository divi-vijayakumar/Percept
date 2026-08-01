"""The claim conversation, end to end — audio in, ONE authoritative value out.

The capability, not a mechanism: a value is asserted, referred to three more times without being
asserted, corrected by an authorized speaker, and disputed by an unauthorized one. Six utterances
reach the judge; four of them contain the string ``$25K``; exactly ONE of them moves the value; and
the last, textually a perfect correction, moves nothing because the speaker may not make it.

Everything below is driven through the real audio runtime — ``percept.runtime.worker.run`` with an
``audio_source``, an acoustic wake gate, a compiled Watch and a scripted judge — so the chain under
test is feed(audio) -> ears -> claim -> delivery, not a state machine called directly. The pure
state machine already has a home (``test_claims.py``) and the internal fold seam has one
(``test_y13_claim_governing.py``); neither can tell you whether a correction spoken into a
microphone reaches a subscriber LINKED to the claim it corrects.

Three things this file measures that a mechanism test structurally cannot:

* **the funnel of the whole run**, read through ``percept.eval.stage_attribution`` rather than by
  counting trace rows by hand. That is where the audio lane's funnel blindness showed up
  (FIND-CLAIMCONV-01, fixed): ``source.audio`` is the admission twin of ``source.frame`` and was
  folded into no counter, so an audio run reported ``observed=0 / admitted=0`` while judging every
  utterance and the dispatch step's attrition went negative.
* **the delivered chain** — incidents carry ``revision_id``/``supersedes``, and walking them is how
  a subscriber learns that the second incident corrects the first rather than contradicting it.
* **the cross-cutting invariants for every stage traversed**: one episode for one conversation, no
  entity id minted or leaked, speaker attribution never inferred from transcript text, and what
  survives a restart.

What it does not establish: that a real model classifies these speech acts, or that real
diarization produces the speaker labels. Both are scripted here, deliberately and visibly — the
labels ride an out-of-band channel precisely so that no part of this test can be passed by reading
the words.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import struct
import tempfile
import unittest
import warnings
from pathlib import Path

from percept import Percept, UndrivenBehaviorRouteWarning
from percept.adapters import Goal, Verdict
from percept.audio_input import (
    AudioInputCapability,
    audio_frame_from_units,
    audio_script_token,
    canonicalize_audio_unit,
)
from percept.backends import anthropic as _anthropic
from percept.backends import gemini as _gemini
from percept.backends import openrouter as _openrouter
from percept.backends.fake import FakeVision
from percept.claims import ClaimState, concern_authority_of, fold_verdicts
from percept.config import GateSettings, Settings
from percept.contracts import AudioFrame, ConcernAuthority, WatchPlan
from percept.events import AudioUnit
from percept.eval.stage_attribution import attribute
from percept.harness.synthetic import generate_all, load_corpus
from percept.runtime.worker import run
from percept.tracing import (
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGE_FINISHED,
    STAGE_JUDGE_STARTED,
    STAGE_SOURCE_AUDIO,
    STAGE_SOURCE_FRAME,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
)
from percept.tunables import REGISTRY as TUNABLES_REGISTRY
from percept.watchplan_apply import apply_watch_plan


# --------------------------------------------------------------------------------------------------
# The audio harness: one utterance == one canonical audio unit with its own bytes.
# --------------------------------------------------------------------------------------------------

CAPABILITY = AudioInputCapability(modes=frozenset({"audio_only"}))

#: Seconds between utterances. Larger than the 4.0s rolling audio window, so each judged turn
#: carries exactly its own unit — the evidence tuple is the utterance, never a mixture of two.
TURN_SPACING_S = 12.0

CONDITION = "a BUYER-side speaker states the approved budget"

#: The value token an utterance CARRIES. Containing it is not asserting it — that is the point.
_MONEY = re.compile(r"\$\d+K")


def _audio_frame(samples: tuple[int, ...], *, start: float, rate: int = 4) -> AudioFrame:
    raw = struct.pack("<" + "h" * len(samples), *samples)
    return AudioFrame(
        pcm=raw,
        sample_rate=rate,
        channels=1,
        sample_format="s16le",
        t_start=start,
        t_end=start + len(samples) / rate,
    )


#: Mean-square energy of an utterance nobody could hear: real signal, an order of magnitude under
#: the acoustic wake bar this file runs the gate at.
INAUDIBLE_ENERGY = (300 / 32768) ** 2


def _unit(seq: int, *, start: float, amplitude: int, energy: float = 1.0) -> AudioUnit:
    """One audio chunk. ``amplitude`` differs per turn so the payload — and therefore the
    scripted-judge key — is unique to the utterance; ``energy`` is what the wake gate reads."""
    chunk = _audio_frame((amplitude,), start=start)
    return AudioUnit(
        seq=seq,
        t_start=chunk.t_start,
        t_end=chunk.t_end,
        event_t=chunk.t_end,
        chunk=chunk,
        energy=energy,
        boundaries=(),
    )


def _canonical(unit: AudioUnit) -> AudioUnit:
    return canonicalize_audio_unit(
        unit,
        capability=CAPABILITY,
        source_id="audio",
        source_epoch=0,
        source_sequence=unit.seq,
        observed_time=10.0,
        source_time=unit.t_end,
    )


def _token_for(unit: AudioUnit) -> str:
    """The FakeVision key for the exact provider payload this unit becomes."""
    return audio_script_token(
        audio_frame_from_units((_canonical(unit),), capability=CAPABILITY)
    )


class Turn:
    """One utterance: what was said, who said it, and the speech act the judge emitted for it.

    ``speaker`` is carried OUT OF BAND (it becomes ``Verdict.asserted_by``). Nothing in the pipeline
    may recover it from ``text`` — see ``SpeakerAttributionIsNeverInferredFromText``.
    """

    def __init__(
        self,
        index: int,
        speaker: str,
        text: str,
        operation: str,
        value: "str | None",
        normalized_value: "int | None" = None,
        audible: bool = True,
        t_start: "float | None" = None,
    ) -> None:
        self.index = index
        self.speaker = speaker
        self.text = text
        self.operation = operation
        self.value = value
        self.normalized_value = normalized_value
        self.audible = audible
        self.t_start = index * TURN_SPACING_S if t_start is None else float(t_start)
        self.unit = _unit(
            index,
            start=self.t_start,
            amplitude=30_000 + 137 * index,
            energy=1.0 if audible else INAUDIBLE_ENERGY,
        )
        #: The event clock instant the runtime stamps this turn's decision with.
        self.event_t = self.unit.t_end

    @property
    def verdict(self) -> Verdict:
        return Verdict(
            True,
            0.95,
            self.text,
            operation=self.operation,
            value=self.value,
            normalized_value=self.normalized_value,
            asserted_by=self.speaker,
        )


def _operation_of(act: str) -> str:
    """A fixture speech-act label as a ``Verdict.operation``. ``negation`` is a ``mention`` — the
    value is referenced and denied, never asserted."""
    return "mention" if act == "negation" else act


class ClaimRunResult:
    """Everything one conversation produced: what reached the user, what the ledgers hold, and the
    trace the stage attribution is derived from."""

    def __init__(self, core, goal_id, incidents, records, manifest, downgrades):
        self.core = core
        self.goal_id = goal_id
        self.incidents = incidents
        self.records = records
        self.manifest = manifest
        self.downgrades = downgrades

    @property
    def episode(self):
        return self.core.episode_current(self.goal_id)

    @property
    def claim_state(self) -> ClaimState:
        return self.core._claim_states[self.goal_id]

    @property
    def attribution(self):
        return attribute(self.records, manifest=self.manifest)


class _ClaimLogCapture(logging.Handler):
    """Capture the structured ``percept.claims`` lines the fold emits, with their fields."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, msg: str) -> list[dict]:
        return [
            dict(getattr(r, "fields", {}) or {}) for r in self.records if r.getMessage() == msg
        ]


def drive_conversation(
    turns: "list[Turn]",
    *,
    authority: "list[str] | None" = ("buyer",),
    goal_id: str = "budget",
    extra_state: "dict | None" = None,
) -> ClaimRunResult:
    """Speak ``turns`` into a compiled claim-slot audio Watch and collect everything it produced."""
    script = {_token_for(turn.unit): turn.verdict for turn in turns}
    if len(script) != len(turns):  # a payload collision would silently re-use one turn's verdict
        raise AssertionError("two turns produced the same audio payload; amplitudes must differ")
    core = Percept.create(vision=FakeVision(script=script), discover_plugins=False)
    receipt = apply_watch_plan(
        core,
        WatchPlan(
            id=goal_id,
            condition=CONDITION,
            sense="audio",
            explicit={"Goal.state.identity_policy": "claim_slot"},
        ),
    )
    if receipt.status != "applied":
        raise AssertionError(f"claim watch did not arm: {receipt}")
    goal = core.goals.get(goal_id)
    if authority is not None:
        # Authored directly on goal.state: the WatchPlan surface cannot carry this — see
        # AuthorityIsUnreachableFromThePortablePlan below.
        goal.state["authority"] = list(authority)
    if extra_state:
        goal.state.update(extra_state)

    recorder = TraceRecorder(
        RunTraceManifest.create(
            source="audio",
            driver="audio-only",
            goals=list(core.goals.active()),
            policy=TracePolicy(level="full"),
        )
    )
    capture = _ClaimLogCapture()
    claims_logger = logging.getLogger("percept.claims")
    claims_logger.addHandler(capture)
    previous_level = claims_logger.level
    claims_logger.setLevel(logging.INFO)
    # Several modules silence logging wholesale for the duration of a class. "Was the downgrade
    # traced?" must not become order-dependent, so lift any process-wide disable for this run only
    # and put back exactly what was there.
    previous_disable = logging.root.manager.disable
    logging.disable(logging.NOTSET)

    async def collect():
        return [
            incident
            async for incident in run(
                core,
                source=None,
                audio_source=[turn.unit for turn in turns],
                audio_settle_s=0.0,
                audio_wake={"high": 0.1, "low": 0.05},
                judge_dispatch="inline",
                on_trace=recorder.tracer(),
            )
        ]

    try:
        incidents = asyncio.run(collect())
    finally:
        logging.disable(previous_disable)
        claims_logger.removeHandler(capture)
        claims_logger.setLevel(previous_level)
    recorder.close()
    return ClaimRunResult(
        core=core,
        goal_id=goal_id,
        incidents=incidents,
        records=list(recorder.records),
        manifest=recorder.manifest,
        downgrades=capture.messages("claim.authority_downgrade"),
    )


# --------------------------------------------------------------------------------------------------
# The corpus is the script. The fixture supplies the utterances and the speech acts; this file
# supplies only the audio and the wiring.
# --------------------------------------------------------------------------------------------------


class _Corpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        generate_all(cls.root)
        cls.corpus = load_corpus("mutation", cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    @classmethod
    def turns_for(cls, case_id: str) -> list[Turn]:
        """The fixture's utterances as spoken turns: its text, its speaker, its labelled act, at its
        own timestamps. The value comes from the utterance and the act from the label — the corpus's
        own text-to-value classification is explicitly not under test (the fixture says so)."""
        case = cls.corpus.case(case_id)
        acts = case.expect["acts"]
        turns = []
        for index, (step, act) in enumerate(zip(case.steps, acts)):
            text = step["text"]
            money = _MONEY.search(text)
            turns.append(
                Turn(
                    index,
                    step["speaker"],
                    text,
                    _operation_of(act),
                    money.group(0) if money else None,
                    t_start=float(step["t"]),
                )
            )
        return turns


#: The disputed final turn. Textually a perfect correction — the same sentence shape the buyer's
#: accepted correction used — from a speaker the concern does not permit to make it.
DISPUTED_TEXT = "Actually, the approved budget is $18K"


class ClaimConversationOverAudio(_Corpus):
    """The whole conversation, once, through the audio runtime — then asserted from many angles."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.case = cls.corpus.case("only_one_of_four_25k_utterances_replaces")
        cls.buyer_turns = cls.turns_for("only_one_of_four_25k_utterances_replaces")
        cls.disputed = Turn(
            len(cls.buyer_turns), "seller", DISPUTED_TEXT, "replace", "$18K",
            t_start=cls.buyer_turns[-1].t_start + TURN_SPACING_S,
        )
        cls.turns = [*cls.buyer_turns, cls.disputed]
        cls.result = drive_conversation(cls.turns)

    # -- the row: containing a value is not asserting it ----------------------------------------
    def test_four_of_five_utterances_contain_the_value_and_one_moves_it(self):
        containing = [t.index for t in self.buyer_turns if "$25K" in t.text]
        self.assertEqual(
            containing,
            self.case.expect["utterances_containing_value"],
            "the premise: the fixture's value-containing turns",
        )
        self.assertEqual(len(containing), 4)

        moved_at = self.case.expect["utterances_mutating_value"]
        self.assertEqual(moved_at, [4], "the fixture says exactly one utterance moves the value")
        mover = self.buyer_turns[moved_at[0]]

        episode = self.result.episode
        # ONE state change after the open, and its instant identifies WHICH utterance made it. The
        # three value-containing turns between the assert and the correction move nothing at all.
        self.assertEqual([r.kind for r in episode.revisions], ["opened", "updated"])
        self.assertEqual(episode.revisions[-1].t, mover.event_t)
        self.assertEqual(episode.revisions[-1].value, "$25K")
        self.assertEqual(episode.value, self.case.expect["final_value"])

    def test_every_utterance_was_judged_but_only_two_reached_the_user(self):
        judged = [r for r in self.result.records if r.stage == STAGE_JUDGE_STARTED]
        created = [r for r in self.result.records if r.stage == STAGE_INCIDENT_CREATED]
        self.assertEqual(len(judged), len(self.turns), "every utterance woke the gate")
        self.assertEqual(len(created), 2, "the establish and the accepted correction, nothing else")
        self.assertEqual(len(self.result.incidents), 2)
        outcomes = self.result.attribution.funnel.gate_outcomes
        self.assertEqual(outcomes["fire"], 2)
        self.assertEqual(outcomes["ask"], 0)
        # The four non-mutating turns are classified and go quiet — a restatement is not an event.
        self.assertEqual(outcomes["silent"], len(self.turns) - 2)

    # -- the row: authority ----------------------------------------------------------------------
    def test_an_unauthorized_replace_moves_nothing_and_reaches_no_one(self):
        # The disputed turn is the LAST one, so anything it moved would be the final state.
        self.assertEqual(self.result.claim_state.value, "$25K")
        self.assertEqual(self.result.episode.value, "$25K")
        self.assertNotIn("$18K", [i.verdict.value for i in self.result.incidents])
        self.assertEqual(
            [r.kind for r in self.result.episode.revisions],
            ["opened", "updated"],
            "a refused replace appends no row — the timeline is not a log of attempts",
        )
        self.assertEqual(self.result.core.metrics.get("gate.superseding_revision"), 1)

    def test_the_downgrade_is_traced_never_silent(self):
        self.assertEqual(len(self.result.downgrades), 1, self.result.downgrades)
        line = self.result.downgrades[0]
        self.assertEqual(line["requested"], "replace")
        self.assertEqual(line["effective"], "mention")
        self.assertEqual(line["asserted_by"], "seller")
        self.assertEqual(line["authorized"], ["buyer"])
        self.assertEqual(line["value"], "$18K")
        self.assertEqual(line["claim_id"], self.result.goal_id)

    # -- the row: supersession -------------------------------------------------------------------
    def test_the_delivered_revisions_name_the_one_they_supersede(self):
        first, second = self.result.incidents
        self.assertIsNone(first.supersedes, "an original claim supersedes nothing")
        self.assertEqual(second.supersedes, first.revision_id)
        self.assertNotEqual(second.revision_id, first.revision_id)
        # Walkable: from the correction, name its predecessor and find it among what was delivered.
        by_revision = {i.revision_id: i for i in self.result.incidents}
        predecessor = by_revision[second.supersedes]
        self.assertEqual(predecessor.verdict.value, "$40K")
        self.assertEqual(second.verdict.value, "$25K")
        self.assertEqual(predecessor.ts, self.buyer_turns[0].event_t)
        self.assertEqual(second.ts, self.buyer_turns[4].event_t)

    def test_the_claim_ledger_agrees_with_the_fixture_revision_chain(self):
        chain = self.case.expect["revision_chain"]
        state = self.result.claim_state
        # The fixture's revision ids are minted from its claim_id, so the two must be the same slot.
        self.assertEqual(self.case.expect["claim_id"], self.result.goal_id)
        self.assertEqual(state.claim_id, self.result.goal_id)
        self.assertEqual(state.revisions, len(chain))
        self.assertEqual(state.revision_id, chain[-1]["revision_id"])
        self.assertEqual(state.value, self.case.expect["final_value"])
        # The normalization the fixture does not carry, but the slot does: it survives the whole
        # conversation including the refused replace.
        self.assertIsNone(state.normalized_value, "no turn in this case stated a canonical form")

    def test_the_judged_verdict_stream_reproduces_the_fixture_trajectory(self):
        """The bridge between the labelled corpus and what the audio lane actually carried: the
        fixture's per-turn ``value_after`` ground truth, folded over the EXACT verdict stream the
        runtime judged, and the live run's own ledger agreeing with the result."""
        state, outcomes = fold_verdicts(
            [turn.verdict for turn in self.buyer_turns],
            claim_id=self.case.expect["claim_id"],
            authority=ConcernAuthority(frozenset({"buyer"})),
            trace=False,
        )
        self.assertEqual([o.value_after for o in outcomes], self.case.expect["value_after"])
        self.assertEqual(state.value, self.case.expect["final_value"])
        supersession = self.case.expect["supersedes"]
        self.assertEqual(
            outcomes[supersession["turn_index"]].supersedes,
            outcomes[supersession["supersedes_turn_index"]].revision_id,
        )
        self.assertEqual(
            outcomes[supersession["supersedes_turn_index"]].value_after,
            supersession["superseded_value"],
        )
        # ...and the live audio run, which additionally heard the disputed seller turn, agrees.
        self.assertEqual(self.result.claim_state.value, state.value)
        self.assertEqual(self.result.claim_state.revision_id, state.revision_id)

    def test_the_superseded_value_is_recoverable_from_the_episode_timeline(self):
        """The chain is walkable from the TIMELINE alone: each row names the value it established.

        Was the characterization for finding claimconv-06. ``observe_decision`` appends the
        ``opened`` row when the episode is born, before ``_claim_episode_fold`` knows what the claim
        machine did with the utterance, so the row that marks the establishment of $40K was written
        with ``value=None`` and nothing back-filled it. A consumer replaying only
        ``episode.revisions`` saw a slot that was empty and became $25K, and the $40K that was
        authoritative for 48 seconds and was DELIVERED as the first incident was nowhere on the
        timeline — including in ``to_judge_dict``, which is what the judge is shown on every later
        tick of the conversation. The fold now names the value on the establishing write, so the
        value a correction superseded is recoverable without replaying the incident stream.
        """
        opened = self.result.episode.revisions[0]
        self.assertEqual(opened.kind, "opened")
        self.assertEqual(opened.value, "$40K", "the establishment is named on the timeline")
        # ...and it is the same value the first incident delivered, so the two ledgers agree.
        self.assertEqual(opened.value, self.result.incidents[0].verdict.value)
        judge_view = self.result.episode.to_judge_dict()
        self.assertEqual([r["value"] for r in judge_view["revisions"]], ["$40K", "$25K"])

    # -- cross-cutting: one occurrence, no identity leak -------------------------------------------
    def test_one_conversation_is_one_episode(self):
        log = self.result.core.episodes[self.result.goal_id]
        self.assertEqual(log.count(), 1, "six utterances about one claim are one occurrence")
        self.assertEqual([s["id"] for s in log.summaries()], [self.result.episode.id])
        self.assertEqual(self.result.episode.subject_key, self.result.goal_id)

    def test_no_entity_id_is_minted_or_delivered(self):
        # An audio claim conversation has no visual subject. Nothing may invent one, and no incident
        # may carry an entity handle a subscriber could join on.
        self.assertEqual(self.result.core.graph.snapshot(), [])
        self.assertEqual([i.entity_id for i in self.result.incidents], [None, None])
        self.assertEqual([i.modality for i in self.result.incidents], ["audio", "audio"])

    def test_delivered_evidence_is_the_utterance_that_was_judged(self):
        # Delivery carries the exact audio the judge saw — one unit per incident, the right one.
        for incident, turn in zip(self.result.incidents, (self.buyer_turns[0], self.buyer_turns[4])):
            self.assertEqual(incident.snippet.kind, "audio")
            self.assertEqual([f.seq for f in incident.snippet.frames], [turn.index])
            self.assertEqual(
                (incident.snippet.t_start, incident.snippet.t_end),
                (turn.t_start, turn.event_t),
            )

    # -- stage attribution ------------------------------------------------------------------------
    def test_stage_attribution_is_quotable_and_names_the_judge_stage(self):
        attribution = self.result.attribution
        self.assertTrue(attribution.publishable, attribution.describe())
        self.assertIn("judge", attribution.measured_segments)
        self.assertIn("delivery", attribution.measured_segments)
        # Both judged spans closed; nothing was truncated by a deadline, so a percentile may be read.
        self.assertGreaterEqual(attribution.percentile("judge", 95.0), 0.0)
        self.assertIsNotNone(attribution.slowest_segment(95.0))

    def test_the_trace_carries_one_admitted_audio_unit_per_utterance(self):
        # The premise of the funnel assertion below: the runtime DOES record every admitted unit.
        source_audio = [r for r in self.result.records if r.stage == STAGE_SOURCE_AUDIO]
        self.assertEqual(len(source_audio), len(self.turns))

    def test_the_funnel_counts_the_audio_units_it_judged(self):
        """FIND-CLAIMCONV-01, fixed (was a strict xfail).

        ``source.audio`` is the admission twin of ``source.frame`` — one record per unit that
        reached the judge — and was folded into no counter, so an audio run reported
        ``observed=0 / admitted=0`` while judging every utterance. The
        ``seen >= gated >= judged >= fired`` spine both ``TraceSummary`` and ``stage_attribution``
        state was violated by an ordinary six-utterance conversation, and admission economy was
        structurally unreportable for audio.
        """
        funnel = self.result.attribution.funnel
        self.assertEqual(funnel.counts["admitted"], len(self.turns))
        self.assertGreaterEqual(funnel.counts["admitted"], funnel.counts["judge_started"])
        self.assertGreaterEqual(funnel.step("dispatch").dropped, 0)
        self.assertIsNotNone(funnel.admission_economy)

    def test_the_funnel_spine_holds_end_to_end_for_an_audio_run(self):
        """The shape the fix restores, stated as the whole spine rather than one step — so this
        cannot rot back into the blindness it replaced. Was the measured characterization of that
        blindness (``admitted=0``, ``dispatch.dropped=-6``, economy ``None``, union ``0.0``)."""
        funnel = self.result.attribution.funnel
        self.assertEqual(funnel.counts["judge_started"], len(self.turns))
        # No `edge.decision` twin exists on the audio lane, so `observed` falls back to `admitted`
        # exactly as it does for a direct perceive_judged() driver — the documented rule, now
        # inheriting a real number rather than a structural zero.
        self.assertEqual(funnel.counts["observed"], len(self.turns))
        spine = [funnel.counts[point] for point in ("observed", "admitted", "judge_started", "fired")]
        self.assertEqual(spine, sorted(spine, reverse=True), funnel.describe())
        for step in funnel.steps:
            self.assertGreaterEqual(step.dropped, 0, step.describe())
        # The admitted utterances covered real seconds of source, so the evidence union is a
        # measurement rather than a structural zero.
        self.assertGreater(funnel.admitted_evidence_union_s, 0.0)


class TheRestOfTheCorpusOverAudio(_Corpus):
    """The other two labelled cases, spoken through the same runtime."""

    def test_a_valueless_retraction_clears_the_value_and_is_delivered_as_a_correction(self):
        case = self.corpus.case("valueless_retraction_clears_the_value")
        turns = self.turns_for("valueless_retraction_clears_the_value")
        self.assertIsNone(turns[-1].value, "the retraction turn names no number at all")
        result = drive_conversation(turns)

        self.assertIsNone(result.episode.value, "cleared to None — absence has one spelling")
        self.assertEqual(result.episode.value, case.expect["final_value"])
        self.assertEqual([r.kind for r in result.episode.revisions], ["opened", "retracted"])
        first, second = result.incidents
        self.assertEqual(second.supersedes, first.revision_id)
        self.assertEqual(result.core.metrics.get("gate.superseding_revision"), 1)

    def test_the_seller_case_resolves_exactly_as_the_fixture_says(self):
        case = self.corpus.case("seller_lacks_authority_to_replace_buyer_value")
        turns = self.turns_for("seller_lacks_authority_to_replace_buyer_value")
        result = drive_conversation(turns, authority=[case.expect["authority_required"]])

        self.assertEqual(result.episode.value, case.expect["final_value"])
        self.assertEqual(len(result.incidents), 1, "the refused replace reaches no one")
        self.assertEqual([r.kind for r in result.episode.revisions], ["opened"])
        blocked = case.expect["blocked_by_authority"]
        self.assertEqual(len(result.downgrades), sum(1 for b in blocked if b))
        self.assertEqual(result.downgrades[0]["asserted_by"], "seller")

    def test_without_a_declared_authority_the_same_conversation_ends_elsewhere(self):
        """The control that proves the authority assertions are not passing for another reason:
        the identical audio, judged identically, with no authority declared, DOES move the value."""
        case = self.corpus.case("seller_lacks_authority_to_replace_buyer_value")
        turns = self.turns_for("seller_lacks_authority_to_replace_buyer_value")
        result = drive_conversation(turns, authority=None)

        self.assertIsNone(concern_authority_of(result.core.goals.get(result.goal_id)))
        self.assertEqual(result.episode.value, "$25K")  # NOT the fixture's $40K
        self.assertNotEqual(result.episode.value, case.expect["final_value"])
        self.assertEqual(result.downgrades, [])
        self.assertEqual(len(result.incidents), 2)


# --------------------------------------------------------------------------------------------------
# The normalized form: an omitted canonical form is not a cleared one, through the live seam.
# --------------------------------------------------------------------------------------------------


class NormalizedFormSurvivesTheConversation(unittest.TestCase):
    """The machine-readable value behind the display text has to survive being restated.

    ``Verdict.normalized_value`` defaults to ``None``, so a judge that restates the agreed number
    without repeating the parse emits exactly the shape that used to destroy it. Asserted here
    through the audio runtime, because the earlier fix lives in the fold and the delivery path is
    what a client actually reads.
    """

    def turns(self) -> list[Turn]:
        return [
            Turn(0, "buyer", "The budget is $25K", "assert", "$25K", 25_000),
            # A deliberate re-assertion of the agreed number, with the canonical form NOT restated.
            Turn(1, "buyer", "Still $25K then", "replace", "$25K", None),
            Turn(2, "buyer", "Yes, $25K as discussed", "confirm", "$25K", None),
        ]

    def test_a_no_op_replace_keeps_the_canonical_form(self):
        result = drive_conversation(self.turns())
        state = result.claim_state
        self.assertEqual(state.value, "$25K")
        self.assertEqual(state.normalized_value, 25_000, "None meant NOT STATED, never 'clear it'")
        self.assertEqual(state.revisions, 1, "restating the same value writes nothing")
        self.assertEqual([r.kind for r in result.episode.revisions], ["opened"])

    def test_a_normalization_only_correction_is_a_real_write(self):
        """The control: the same call shape whose canonical form genuinely differs IS a write, so
        the assertion above is not passing merely because nothing can ever move a normalization."""
        turns = [
            Turn(0, "buyer", "The budget is $25K", "assert", "$25K", 2_500),  # a bad parse
            Turn(1, "buyer", "The budget is $25K", "replace", "$25K", 25_000),  # the fix
        ]
        result = drive_conversation(turns)
        state = result.claim_state
        self.assertEqual(state.normalized_value, 25_000)
        self.assertEqual(state.revisions, 2)
        self.assertEqual([r.kind for r in result.episode.revisions], ["opened", "updated"])
        second = result.incidents[-1]
        self.assertEqual(second.supersedes, result.incidents[0].revision_id)


class ARestatementIsNotDeliveredAsACorrection(unittest.TestCase):
    """A speaker re-affirming the agreed number is agreement, and reaches the subscriber as nothing.

    Was finding claimconv-03, where the two ledgers disagreed and only an end-to-end read could see
    it. The claim ledger was right: nothing was applied, no revision was minted, no row joined the
    timeline. The DELIVERY path fired anyway, because the fold handed ``fold_episode`` the effective
    operation whether or not the write landed, and ``replace`` maps to ``updated`` — which re-arms,
    fires, and links the new incident to a predecessor it never replaced.
    ``Percept._claim_episode_operation`` now folds an act that wrote nothing as the re-statement it
    is, so the delivery path and the claim ledger say the same thing.

    ``test_claims.py::IdenticalValueReplaceTests`` pins the same asymmetry inside the state machine
    and names the sharp edge; what it cannot see is whether the edge reaches a human.
    """

    @staticmethod
    def turns() -> list[Turn]:
        return [
            Turn(0, "buyer", "The budget is $25K", "assert", "$25K", 25_000),
            Turn(1, "buyer", "Still $25K then", "replace", "$25K", None),
        ]

    def test_what_a_restatement_delivers(self):
        """Was the characterization for finding claimconv-03 — TWO incidents for one settled value,
        the second flagged as superseding the first, both carrying ``$25K``, and
        ``gate.superseding_revision`` counting a correction that never happened. Now the two ledgers
        agree: one incident, superseding nothing, and the timeline is exactly what it was."""
        result = drive_conversation(self.turns())
        (delivered,) = result.incidents
        self.assertIsNone(delivered.supersedes, "there was nothing to supersede")
        self.assertEqual(delivered.verdict.value, "$25K")
        self.assertEqual(result.core.metrics.get("gate.superseding_revision"), 0)
        # ...and the claim ledger, which was always right, is unchanged by the fix.
        self.assertEqual(result.claim_state.revisions, 1)
        self.assertEqual([r.kind for r in result.episode.revisions], ["opened"])

    def test_a_restatement_is_not_delivered_as_a_correction(self):
        result = drive_conversation(self.turns())
        self.assertEqual(len(result.incidents), 1, "nothing changed, so nothing was corrected")
        self.assertEqual(result.core.metrics.get("gate.superseding_revision"), 0)

    def test_a_real_correction_in_the_same_shape_does_deliver(self):
        """The control: identical call shape, a genuinely different value, two incidents linked."""
        turns = self.turns()
        turns[1] = Turn(1, "buyer", "Actually the approved budget is $30K", "replace", "$30K")
        result = drive_conversation(turns)
        self.assertEqual(len(result.incidents), 2)
        self.assertEqual(result.incidents[1].supersedes, result.incidents[0].revision_id)
        self.assertEqual(result.claim_state.revisions, 2)
        self.assertEqual([r.kind for r in result.episode.revisions], ["opened", "updated"])


class AMentionDeliversAValueTheLedgerRefused(unittest.TestCase):
    """The headline row again, at the delivery boundary — where it did not hold. Finding claimconv-07.

    "Containing a value is not asserting it" was true of the LEDGER and false of what reached a
    subscriber. The corpus hid it: every labelled case opens with the establishing ``assert``, so
    the first fire and the first write are the same utterance and no one can tell which one caused
    the delivery. Put a value-bearing question FIRST — the ordinary shape of a negotiation, and the
    fixture's own ``Could we do $25K?`` — and the two came apart:

    * the mention fired the episode's first incident, carrying ``$25K``, a value the claim channel
      explicitly refused; the authoritative slot was still empty at that instant;
    * the ``assert`` that followed really did establish ``$40K`` and mint revision one — and folded as
      ``confirmed``, so the already-emitted episode never re-entered the gate and the establishment
      reached no one.

    A subscriber was therefore told ``$25K`` and never told ``$40K``, while ``episode_current`` said
    ``$40K``. The fix takes the disjunction's first branch: an act that does not WRITE the slot
    (``percept.claims.VALUE_SETTING`` — ``mention``/``confirm``) may open an episode but may not open
    its DELIVERY. The episode still opens, ``emitted`` stays 0, and the establishing act that follows
    finds it un-emitted and fires carrying the value the slot actually holds. Nothing outside the
    claim channel threads an ``operation``, so no other fold moves.
    """

    @staticmethod
    def mention_first() -> list[Turn]:
        return [
            Turn(0, "buyer", "Could we do $25K?", "mention", "$25K"),
            Turn(1, "buyer", "The budget is $40K", "assert", "$40K", 40_000),
        ]

    def test_a_mention_into_an_empty_slot_establishes_nothing(self):
        """The ledger half, which is correct — and the case the corpus never exercises, because a
        mention there only ever meets an ALREADY-established slot."""
        result = drive_conversation([Turn(0, "buyer", "Could we do $25K?", "mention", "$25K", 25_000)])
        self.assertIsNone(result.claim_state.value, "a question establishes nothing")
        self.assertIsNone(result.claim_state.normalized_value)
        self.assertEqual(result.claim_state.revisions, 0)
        self.assertIsNone(result.claim_state.revision_id)
        self.assertIsNone(result.episode.value)
        self.assertEqual([r.kind for r in result.episode.revisions], ["opened"])

    def test_what_a_mention_first_conversation_delivers(self):
        """REWRITTEN with the fix for finding claimconv-07 (was the characterization of the wrong
        value). Still exactly one incident for this conversation — the mention no longer manufactures
        one — and it is the ESTABLISHING turn, at that turn's timestamp, carrying the slot's value."""
        result = drive_conversation(self.mention_first())
        self.assertEqual(len(result.incidents), 1)
        delivered = result.incidents[0]
        self.assertEqual(delivered.verdict.value, "$40K")
        self.assertEqual(delivered.verdict.operation, "assert")
        self.assertEqual(delivered.ts, self.mention_first()[1].event_t)
        # ...and it agrees with the ledger and the episode instead of contradicting them.
        self.assertEqual(result.claim_state.value, "$40K")
        self.assertEqual(result.claim_state.revisions, 1)
        self.assertEqual(result.episode.value, "$40K")
        # An original, not a correction: nothing was superseded, because nothing was delivered first.
        self.assertIsNone(delivered.supersedes)
        self.assertEqual(result.core.metrics.get("gate.superseding_revision"), 0)

    def test_the_withheld_opening_is_counted_apart_from_idempotency(self):
        """The withholding is not silent, and it is not filed as "already said". Both suppress before
        the gate, and reading one as the other is how a missing establishing turn would look like
        correct de-duplication in the metrics."""
        result = drive_conversation(self.mention_first())
        metrics = result.core.metrics.snapshot()
        self.assertEqual(metrics.get("gate.unestablished_withheld"), 1)
        self.assertNotIn("gate.idempotent_suppressed", metrics)

    def test_a_mention_alone_delivers_nothing_at_all(self):
        """The narrow form. Nothing is ever asserted, so the slot stays empty — and a subscriber is
        told nothing, rather than told a number the claim channel refused."""
        result = drive_conversation([Turn(0, "buyer", "Could we do $25K?", "mention", "$25K")])
        self.assertIsNone(result.claim_state.value)
        self.assertEqual(result.incidents, [])

    def test_an_incident_still_carries_no_authoritative_claim_value_field(self):
        """The structural half of the old characterization, which the fix does NOT close: the
        delivered object has no field naming the slot's value. ``verdict.value`` is the UTTERANCE's
        value and is still the only one there is — it now agrees with the slot because the utterance
        delivered IS the one that wrote it, not because a client can read the ledger off an incident.
        A conversation where the two must differ still has nowhere to put the difference."""
        result = drive_conversation(self.mention_first())
        delivered = result.incidents[0]
        self.assertEqual(delivered.verdict.value, result.episode.value)
        self.assertEqual(
            [
                name
                for name in (f.name for f in dataclasses.fields(delivered))
                if "value" in name
            ],
            [],
        )

    def test_the_delivering_and_the_writing_vocabularies_do_not_drift(self):
        """``behavior._ESTABLISHING_OPERATIONS`` restates ``claims.VALUE_SETTING`` because ``claims``
        imports from ``behavior`` and the reverse edge would be a cycle. Restated, therefore pinned:
        the acts that WRITE the slot and the acts that may OPEN a delivery are one set, and the
        finding was exactly what happens when the delivery path answers from a different one."""
        from percept.behavior import _ESTABLISHING_OPERATIONS
        from percept.claims import VALUE_SETTING

        self.assertEqual(set(_ESTABLISHING_OPERATIONS), set(VALUE_SETTING))

    def test_the_delivered_value_is_the_value_the_slot_holds(self):
        result = drive_conversation(self.mention_first())
        # Deliberately a disjunction: either the mention must not deliver a value the slot refused,
        # or the establishing assert must reach the user too. This file does not prescribe which.
        delivered_values = [i.verdict.value for i in result.incidents]
        self.assertIn(
            result.episode.value,
            delivered_values,
            f"the slot settled on {result.episode.value!r} and delivered {delivered_values!r}",
        )

    def test_the_same_conversation_in_the_other_order_is_correct(self):
        """The control that makes this a defect rather than a property of the harness: swap the two
        utterances and the delivered value is the slot's value. Nothing else changes."""
        turns = [
            Turn(0, "buyer", "The budget is $40K", "assert", "$40K", 40_000),
            Turn(1, "buyer", "Could we do $25K?", "mention", "$25K"),
        ]
        result = drive_conversation(turns)
        self.assertEqual(len(result.incidents), 1)
        self.assertEqual(result.incidents[0].verdict.value, "$40K")
        self.assertEqual(result.episode.value, "$40K")
        self.assertIn(result.episode.value, [i.verdict.value for i in result.incidents])


# --------------------------------------------------------------------------------------------------
# Cross-cutting: speaker attribution is never inferred from what was said.
# --------------------------------------------------------------------------------------------------


class SpeakerAttributionIsNeverInferredFromText(unittest.TestCase):
    """Who spoke is an out-of-band fact. No part of the chain may reconstruct it from the words.

    The failure this forbids is concrete: a transcript that says "the seller said the budget is
    $25K" must not become a verdict attributed to the seller, because the value channel is
    authority-gated and an inferred speaker is an inferred permission.
    """

    def test_the_three_shipped_backends_share_one_parser(self):
        # There is exactly ONE place a reply could ever grow a speaker; naming it here means the
        # assertion below covers all three rather than one.
        self.assertIs(_gemini._parse_verdict, _anthropic._parse_verdict)
        self.assertIs(_openrouter._parse_verdict, _anthropic._parse_verdict)

    def test_the_parser_refuses_a_speaker_even_when_the_reply_offers_one(self):
        reply = (
            '{"satisfied": true, "confidence": 0.9, '
            '"evidence": "the seller said: actually, the approved budget is $25K", '
            '"operation": "replace", "value": "$25K", "asserted_by": "seller"}'
        )
        verdict = _anthropic._parse_verdict(reply, claim_mode=True)
        self.assertEqual(verdict.operation, "replace")  # the act IS parsed...
        self.assertEqual(verdict.value, "$25K")
        self.assertIsNone(verdict.asserted_by, "...and the speaker is never taken, offered or not")

    def test_a_declared_authority_refuses_every_correction_a_shipped_judge_can_produce(self):
        """The consequence of that refusal, stated as the capability bound it is (finding
        claimconv-05). With no diarization behind the verdict, ``asserted_by`` is ``None``, and
        ``ConcernAuthority.permits`` requires a NAMED asserter for a mutating act. So a concern that
        declares an authority can never accept a correction from any shipped backend."""
        verdict = _anthropic._parse_verdict(
            '{"satisfied": true, "confidence": 0.9, "evidence": "actually $25K", '
            '"operation": "replace", "value": "$25K"}',
            claim_mode=True,
        )
        self.assertIsNone(verdict.asserted_by)
        state = ClaimState(claim_id="budget", value="$40K", revisions=1, revision_id="budget#r1")
        outcome = state.apply(
            verdict.operation,
            value=verdict.value,
            asserted_by=verdict.asserted_by,
            authority=ConcernAuthority(frozenset({"buyer"})),
            trace=False,
        )
        self.assertTrue(outcome.downgraded)
        self.assertEqual(state.value, "$40K", "the value can never be corrected from a live judge")
        # Undeclared authority is the documented escape hatch, and it still works.
        unrestricted = ClaimState(claim_id="budget", value="$40K", revisions=1, revision_id="budget#r1")
        unrestricted.apply(verdict.operation, value=verdict.value, asserted_by=None, trace=False)
        self.assertEqual(unrestricted.value, "$25K")

    def test_relabelling_the_speaker_alone_flips_the_outcome(self):
        """The decisive mutation. Byte-identical audio, byte-identical transcript, byte-identical
        speech act — only the out-of-band label differs, and the value moves. If any component read
        the speaker off the text this could not happen."""
        base = [
            Turn(0, "buyer", "The budget is $40K", "assert", "$40K"),
            Turn(1, "seller", DISPUTED_TEXT, "replace", "$18K"),
        ]
        refused = drive_conversation(base)
        self.assertEqual(refused.episode.value, "$40K")

        relabelled = [
            Turn(0, "buyer", "The budget is $40K", "assert", "$40K"),
            Turn(1, "buyer", DISPUTED_TEXT, "replace", "$18K"),  # same words, different speaker
        ]
        accepted = drive_conversation(relabelled)
        self.assertEqual(accepted.episode.value, "$18K")
        # The audio payload really was identical between the two runs.
        self.assertEqual(_token_for(base[1].unit), _token_for(relabelled[1].unit))
        self.assertEqual(base[1].text, relabelled[1].text)
        self.assertEqual(len(refused.incidents), 1)
        self.assertEqual(len(accepted.incidents), 2)


class AuthorityIsReachableFromThePortablePlan(unittest.TestCase):
    """A claim Watch armed the way the product arms Watches can declare who may correct it.

    Was finding claimconv-04: ``Goal.state.authority`` was absent from ``percept.tunables.REGISTRY``
    while its sibling ``Goal.state.identity_policy`` was present, so ``apply_watch_plan`` rejected
    any plan naming it (``ownership_violation``) and the only way to restrict a claim was to reach
    into ``goal.state`` after the plan applied — which a portable plan crossing a process boundary
    cannot do. Half the claim contract crossed the boundary and the half deciding WHO MAY WRITE did
    not, so a compiled claim Watch always armed unrestricted.

    Read with ``SpeakerAttributionIsNeverInferredFromText`` below, which states the capability bound
    that now applies to anyone who authors it: the authority check reads ``Verdict.asserted_by``,
    which no shipped judge populates.
    """

    def test_an_unknown_state_path_is_still_rejected_outright(self):
        """The mutation guard on the fix: the registry gate is what rejected the authority plan, and
        it is still live. ``Goal.state.authority`` moved to the legal side of it by being registered
        as the real policy it is — the gate itself was not loosened."""
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="budget",
                condition=CONDITION,
                sense="audio",
                explicit={
                    "Goal.state.identity_policy": "claim_slot",
                    "Goal.state.who_may_correct": ["buyer"],
                },
            ),
        )
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(receipt.watch_ids, ())
        dropped = [d.path for d in receipt.dropped_fields]
        self.assertIn("WatchPlan", dropped)
        self.assertIn(
            "Goal.state.who_may_correct",
            " ".join(d.message for d in receipt.dropped_fields),
        )

    def test_the_identity_policy_half_of_claim_mode_is_authorable(self):
        # The half that was never in question — kept as the control the asymmetry was measured
        # against, so a regression that un-registers either half is caught on both sides.
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="budget",
                condition=CONDITION,
                sense="audio",
                explicit={"Goal.state.identity_policy": "claim_slot"},
            ),
        )
        self.assertEqual(receipt.status, "applied")
        self.assertEqual(core.goals.get("budget").state["identity_policy"], "claim_slot")
        self.assertEqual(core._judge_health_log, [], "declaring nothing announces nothing")

    def test_a_plan_can_declare_who_may_correct_the_claim(self):
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="budget",
                condition=CONDITION,
                sense="audio",
                explicit={
                    "Goal.state.identity_policy": "claim_slot",
                    "Goal.state.authority": ["buyer"],
                },
            ),
        )
        self.assertEqual(receipt.status, "applied")
        self.assertEqual(receipt.dropped_fields, ())
        # The authority reaches the ARMED goal, not merely the receipt: the runtime reads it from
        # goal.state at fold time, through the same accessor a hand-built goal goes through.
        self.assertEqual(core.goals.get("budget").state["authority"], ["buyer"])
        self.assertEqual(
            concern_authority_of(core.goals.get("budget")),
            ConcernAuthority(frozenset({"buyer"})),
        )

    def test_declaring_an_authority_announces_that_no_judge_can_satisfy_it(self):
        """Containment for the 04+05 pair. Making the row authorable also makes it PUBLIC — a
        ``visibility='public'`` spec renders into the generated tunables reference, which would be
        the first documentation ``ConcernAuthority`` has ever had. And the knob is inert-and-total
        when set: ``permits`` needs a named ``asserted_by`` and no shipped backend produces one, so
        the value it guards freezes for everyone. The bound is therefore stated ONCE at arm time, on
        the run-level health queue the worker drains — not only as an INFO downgrade line after each
        correction it has already refused.
        """
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="budget",
                condition=CONDITION,
                sense="audio",
                explicit={
                    "Goal.state.identity_policy": "claim_slot",
                    "Goal.state.authority": ["buyer"],
                },
            ),
        )
        self.assertEqual(receipt.status, "applied", "the Watch still arms — this is a bound, not a veto")
        (kind, key, cause), = core._judge_health_log
        self.assertEqual(kind, "claim_authority_unsatisfiable")
        self.assertEqual(key, "budget")
        self.assertIn("asserted_by", cause)
        self.assertIn("never be corrected", cause)
        self.assertEqual(core.metrics.get("goal.armed.claim_authority_unsatisfiable"), 1)

    def test_a_deployment_that_supplies_the_speaker_arms_quietly(self):
        """The escape hatch is a DECLARATION, never an inference — which is also the mutation that
        shows the announcement above is conditional rather than unconditional noise. An adapter that
        states it populates ``Verdict.asserted_by`` (a diarizer, or a client-asserted identity)
        silences it; nothing about the judge is guessed."""
        vision = FakeVision()
        vision.supports_speaker_attribution = True
        core = Percept.create(vision=vision, discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="budget",
                condition=CONDITION,
                sense="audio",
                explicit={
                    "Goal.state.identity_policy": "claim_slot",
                    "Goal.state.authority": ["buyer"],
                },
            ),
        )
        self.assertEqual(receipt.status, "applied")
        self.assertEqual(core._judge_health_log, [])

    def test_the_registry_row_states_the_bound_an_author_of_it_inherits(self):
        """Authoring authority is now possible, and doing so is a capability decision, not a tuning
        one: the check is against ``Verdict.asserted_by``, which no shipped judge populates (see
        ``SpeakerAttributionIsNeverInferredFromText``). The registry row an author reads before
        setting the field has to say so — a knob whose documented effect is the opposite of its
        actual effect on a shipped judge is worse than an absent one."""
        spec = {s.path: s for s in TUNABLES_REGISTRY}["Goal.state.authority"]
        self.assertEqual(spec.owner, "WatchPlan")
        self.assertEqual(spec.visibility, "public")
        self.assertFalse(spec.safe_to_optimize, "who may write is not an optimizer's decision")
        self.assertIn("asserted_by", spec.note)
        self.assertIn("CAPABILITY BOUND", spec.note)


# --------------------------------------------------------------------------------------------------
# Cross-cutting: what a restart does to the claim timeline.
# --------------------------------------------------------------------------------------------------


def _plain_claim_core(script: dict) -> Percept:
    """A claim-slot concern on a plain-goal core — the only shape ``snapshot()`` will serialize."""
    settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=10_000.0))
    core = Percept.create(
        vision=FakeVision(script=script), settings=settings, discover_plugins=False
    )
    core.add_goal(
        Goal(
            id="budget",
            condition=CONDITION,
            say="budget claimed",
            state={
                "type": "watch",
                "count": 0,
                "identity_policy": "claim_slot",
                "authority": ["buyer"],
            },
        )
    )
    return core


ESTABLISH = Verdict(
    True, 0.9, "The budget is $40K", operation="assert", value="$40K",
    normalized_value=40_000, asserted_by="buyer",
)
CORRECT = Verdict(
    True, 0.9, "Actually the approved budget is $25K", operation="replace", value="$25K",
    normalized_value=25_000, asserted_by="buyer",
)


class TheClaimTimelineAcrossARestart(unittest.TestCase):
    """A conversation outlives a worker. The claim it settled has to outlive one too — or say so."""

    def test_a_compiled_claim_watch_refuses_to_produce_a_partial_snapshot(self):
        # The honest half. A compiled Watch knows its episode state is session-only and REFUSES to
        # hand back a recovery artifact that would silently omit it.
        turns = [Turn(0, "buyer", "The budget is $40K", "assert", "$40K", 40_000)]
        result = drive_conversation(turns)
        with self.assertRaises(Exception) as caught:
            result.core.snapshot()
        self.assertIn("session-only", str(caught.exception))

    def _restarted(self):
        """Establish $40K, snapshot, restore into a FRESH core (a restart is a new process), then
        speak the correction. Returns the two cores, what crossed the restart, and what the
        correction produced."""
        before = _plain_claim_core({"turn-0": ESTABLISH})
        asyncio.run(before.perceive_judged("turn-0", t=0.0))
        snapshot = before.snapshot()

        after = _plain_claim_core({"turn-1": CORRECT})
        after.restore(snapshot)
        # Read the recovered state BEFORE the next utterance touches it: this is what the restarted
        # worker woke up holding.
        recovered = (dict(after.episodes), dict(after._claim_states))
        events = asyncio.run(after.perceive_judged("turn-1", t=25.0))
        return before, after, recovered, events

    def test_what_a_restart_actually_does_to_the_claim(self):
        """The half of finding claimconv-02 that is ratified policy, not a defect: episodes and the
        claim ledger are SESSION-ONLY (ADR-006), so neither crosses the restart. Recorded so the
        disjunction below cannot rot into passing on the wrong half — it is satisfied by the
        ANNOUNCEMENT, and this is what is being announced.

        What the correction then DOES is owned by the restart lane, not by this file. The fix for
        findings restart-01/02/03 makes a gate key read back from a snapshot refuse to be re-armed
        while its edge is spent — which for a moment dropped this correction entirely, and silence is
        worse than wrong linkage: the user says "actually it is $25K", the ledger records it, and
        nobody is told. The refusal is therefore conditioned on the SPEECH ACT
        (``EpisodeFoldResult.fact_changed``): a re-statement of a still-running occurrence is refused,
        an authority-gated ``replace``/``retract``/``close`` is honoured. So the correction arrives —
        as an ORIGINAL claim, ``supersedes=None`` and ``revision_id`` back at #r1. The $40K it
        replaced is nowhere in the output and nothing links the two, which is precisely what the
        health event has to say out loud.
        """
        before, after, recovered, events = self._restarted()
        self.assertEqual(before.episode_current("budget").value, "$40K")
        self.assertEqual(before._claim_states["budget"].revision_id, "budget#r1")

        # Nothing about the claim crossed the restart: not the episode, not the ledger.
        self.assertEqual(recovered, ({}, {}))

        # The correction re-establishes the slot from scratch — revision ONE again, superseding
        # nothing — on a new episode whose timeline knows nothing of the value it replaced.
        self.assertEqual(
            [e.action for e in events], ["fire"],
            "a correction after a restart still reaches the subscriber — silence is not the fix",
        )
        (delivered,) = events
        self.assertEqual(delivered.revision_id, "budget:ep1#r1", "revision ONE again")
        self.assertIsNone(delivered.supersedes, "…and linked to nothing, which is the finding")
        self.assertEqual(after.metrics.get("gate.superseding_revision"), 0)
        self.assertEqual(after._claim_states["budget"].revisions, 1)
        self.assertEqual(after._claim_states["budget"].value, "$25K")
        self.assertEqual(
            [(r.kind, r.value) for r in after.episode_current("budget").revisions],
            [("opened", "$25K")],
            "the $40K it superseded is not on the new timeline either",
        )

    def test_the_amnesia_reaches_the_health_channel(self):
        """The part that made it a defect rather than a documented boundary, now closed.

        ``restore`` queues ``episode_memory_reset`` precisely so the session-only loss is observable,
        and a claim-slot concern used to be in neither of its two sets (``type='count'`` and
        ``rearm='each_event'``) — so a claim lost its value, its revision chain and its supersession
        link in silence. The predicate now also covers every concern that SELECTS an identity policy,
        which is the same resolution the subject-bound episode folds themselves gate on.
        """
        _before, after, _recovered, _events = self._restarted()
        self.assertEqual([kind for kind, _ in after._restore_health_log], ["episode_memory_reset"])
        (_kind, data), = after._restore_health_log
        self.assertEqual(data["goal_ids"], ["budget"])
        self.assertEqual(data["subject_bound_goals"], 1)
        self.assertEqual(data["counting_goals"], 0, "it is not announced as a counting concern")
        self.assertEqual(data["occurrence_goals"], 0)

    def test_the_health_channel_itself_works_for_a_counting_concern(self):
        """The control the claim-slot case was measured against: the mechanism was always live."""
        counting = Goal(
            id="cups", condition="a cup is filled", say="cup", state={"type": "count", "count": 0}
        )
        with warnings.catch_warnings():
            # A counting concern is driven by perceive_boundary, not by frames; the runtime says so
            # on arming and this test never drives it.
            warnings.simplefilter("ignore", UndrivenBehaviorRouteWarning)
            core = _plain_claim_core({})
            core.add_goal(counting)
            snapshot = core.snapshot()
            restored = _plain_claim_core({})
            restored.add_goal(counting)
            restored.restore(snapshot)
        kinds = [kind for kind, _data in restored._restore_health_log]
        self.assertEqual(kinds, ["episode_memory_reset"])

    def test_a_restart_either_keeps_the_claim_or_says_it_did_not(self):
        # Deliberately a disjunction: either fix closes it, and this file does not prescribe which.
        _before, after, _recovered, events = self._restarted()
        survived = after._claim_states.get("budget") is not None and any(
            e.supersedes is not None for e in events
        )
        announced = any(kind == "episode_memory_reset" for kind, _ in after._restore_health_log)
        self.assertTrue(
            survived or announced,
            "the claim timeline was lost across a restart with nothing said about it",
        )


# --------------------------------------------------------------------------------------------------
# Mutation guards — every bar above is shown to be capable of failing.
# --------------------------------------------------------------------------------------------------


class TheBarsCanFail(_Corpus):
    """A gate that has never failed is not evidence. Each test here perturbs exactly one thing the
    assertions above depend on and shows the outcome moves."""

    def test_scripting_the_correction_as_a_mention_collapses_the_chain(self):
        # Perturbs the speech act only. If supersession were falling out of the gate rather than
        # out of the claim channel, this would still deliver two linked incidents.
        turns = self.turns_for("only_one_of_four_25k_utterances_replaces")
        turns[4] = Turn(4, "buyer", turns[4].text, "mention", "$25K", t_start=turns[4].t_start)
        result = drive_conversation(turns)
        self.assertEqual(result.episode.value, "$40K", "a mention moves nothing, however it reads")
        self.assertEqual([r.kind for r in result.episode.revisions], ["opened"])
        self.assertEqual(len(result.incidents), 1)
        self.assertEqual(result.core.metrics.get("gate.superseding_revision"), 0)

    def test_shrinking_the_continuity_gap_splits_the_conversation_into_many_episodes(self):
        # Proves the "one conversation is one episode" bar counts something real: the same six
        # utterances, with the concern's continuity gap authored below the spacing between them,
        # become an episode each. A constant-1 assertion could not move.
        turns = self.turns_for("only_one_of_four_25k_utterances_replaces")
        intact = drive_conversation(turns)
        self.assertEqual(intact.core.episodes[intact.goal_id].count(), 1)

        split = drive_conversation(
            turns, extra_state={"episode": {"continuity_gap_s": TURN_SPACING_S / 4.0}}
        )
        self.assertGreater(split.core.episodes[split.goal_id].count(), 1)

    def test_an_utterance_under_the_wake_bar_never_reaches_the_claim_at_all(self):
        """The ears really are in this chain. Drop one turn's energy below the acoustic wake bar —
        same words, same speech act, same speaker — and it is never judged, never folded, and never
        delivered. Without this, "feed(audio) -> ears" would be decoration."""
        turns = self.turns_for("only_one_of_four_25k_utterances_replaces")
        turns[4] = Turn(
            4, "buyer", turns[4].text, "replace", "$25K",
            audible=False, t_start=turns[4].t_start,
        )
        result = drive_conversation(turns)

        judged = [r for r in result.records if r.stage == STAGE_JUDGE_STARTED]
        self.assertEqual(len(judged), len(turns) - 1, "the inaudible utterance woke nothing")
        self.assertEqual(result.episode.value, "$40K", "so it corrected nothing")
        self.assertEqual([r.kind for r in result.episode.revisions], ["opened"])
        self.assertEqual(len(result.incidents), 1)

    def test_the_audio_admission_stage_is_what_the_funnel_counts(self):
        """The counted thing is ``source.audio`` ITSELF, not something that happens to co-occur.

        This was the achievability proof for the funnel pin: relabelling exactly those rows to the
        video lane's admission stage — touching nothing else about the run — turned a structurally
        blind funnel into a whole one, which is what showed the gap was a missing fold rather than a
        missing record. With the fold landed the relabel is now a NO-OP, and that is the assertion:
        the audio lane and the video lane are counted identically. The mutation below is the control
        that keeps this honest — DELETE those rows and the funnel loses exactly them.
        """
        turns = self.turns_for("only_one_of_four_25k_utterances_replaces")
        result = drive_conversation(turns)
        funnel = result.attribution.funnel
        self.assertEqual(funnel.counts["admitted"], len(turns))
        self.assertGreaterEqual(funnel.counts["admitted"], funnel.counts["judge_started"])
        self.assertGreaterEqual(funnel.step("dispatch").dropped, 0)
        self.assertIsNotNone(funnel.admission_economy)

        relabelled = attribute(
            [
                dataclasses.replace(r, stage=STAGE_SOURCE_FRAME)
                if r.stage == STAGE_SOURCE_AUDIO
                else r
                for r in result.records
            ],
            manifest=result.manifest,
        )
        self.assertEqual(dict(relabelled.funnel.counts), dict(funnel.counts))

        # The control: without those rows there is nothing to count, so the count really is theirs.
        stripped = attribute(
            [r for r in result.records if r.stage != STAGE_SOURCE_AUDIO],
            manifest=result.manifest,
        )
        self.assertEqual(stripped.funnel.counts["admitted"], 0)

    def test_the_entity_check_reports_a_graph_that_has_something_in_it(self):
        # "No entity was minted" is only evidence if the same read would have shown one. Ingest a
        # single observation into the very core the conversation ran on and the graph speaks up.
        from percept.contracts import Box, Observation

        turns = self.turns_for("only_one_of_four_25k_utterances_replaces")
        result = drive_conversation(turns)
        self.assertEqual(result.core.graph.snapshot(), [])
        result.core.graph.ingest(
            [Observation(label="person", box=Box(0.0, 0.0, 1.0, 1.0), confidence=0.9)], t=0.0
        )
        self.assertEqual(len(result.core.graph.snapshot()), 1)

    def test_removing_the_incident_rows_moves_the_attribution(self):
        # The stage-attribution helper must report a DIFFERENT table for a different trace. A helper
        # that answers the same for both is worse than none.
        turns = self.turns_for("only_one_of_four_25k_utterances_replaces")
        result = drive_conversation(turns)
        full = result.attribution
        self.assertEqual(full.funnel.counts["fired"], 2)
        self.assertIn("delivery", full.measured_segments)

        mutant = [r for r in result.records if r.stage != STAGE_INCIDENT_CREATED]
        moved = attribute(mutant, manifest=result.manifest)
        self.assertEqual(moved.funnel.counts["fired"], 0)
        self.assertNotIn("delivery", moved.measured_segments)
        self.assertNotIn("end_to_end", moved.measured_segments)

    def test_an_unclosed_judge_span_makes_the_percentile_unquotable(self):
        # The refusal is the feature: a sample that excludes the calls which never finished must not
        # be quoted. Drop the closing rows for one judged turn and the segment stops being sayable.
        turns = self.turns_for("only_one_of_four_25k_utterances_replaces")
        result = drive_conversation(turns)
        self.assertTrue(result.attribution.is_publishable("judge"))

        dropped_one = 0
        mutant = []
        for record in result.records:
            if record.stage == STAGE_JUDGE_FINISHED and dropped_one == 0:
                dropped_one = 1
                continue
            mutant.append(record)
        self.assertEqual(dropped_one, 1)
        censored = attribute(mutant, manifest=result.manifest)
        self.assertFalse(censored.is_publishable("judge"))
        with self.assertRaises(Exception):
            censored.percentile("judge", 95.0)


if __name__ == "__main__":
    unittest.main()
