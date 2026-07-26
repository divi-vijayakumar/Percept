"""— subject / continuation / mutation are THREE decisions, not one.

The acceptance criteria come from the external review's decisive example: PO-4471 is the same
SUBJECT every time it appears; it is NOT necessarily the same EPISODE. Exact subject equality is
NECESSARY EVIDENCE for continuation, never its definition::

    continues = (same_subject
                 and episode.status == "open"
                 and observation.ts - episode.last_seen <= continuity_gap_s
                 and mutation_is_compatible)

The stream cases (inside-gap continues / after-gap opens new / A-B-A no spurious third fire /
terminal-reopen) are driven from ``eval/synthetic/continuation/cases.json`` when that fixture tree is
present, and from the identical embedded copy when it is not — the assertions are the acceptance
criteria either way.
"""

import json
import unittest
from pathlib import Path

from percept.adapters import Goal
from percept.agent import Percept
from percept.behavior import BehaviorProgram, Accumulator, BehaviorExecutor, coerce_authored_episode
from percept.episodes import (
    CLOSED,
    CONTINUATION_REASONS,
    EPISODE_POLICIES,
    EpisodeLog,
    EpisodePolicy,
    episode_policy_of,
)
from percept.gate import Gate

_FIXTURE = (
    Path(__file__).resolve().parents[3] / "eval" / "synthetic" / "continuation" / "cases.json"
)

# Identical to the fixture — used when the synthetic tree is absent (a worktree without eval/).
_EMBEDDED = {
    "cases": [
        {
            "name": "inside_gap_continues",
            "policy": {"continuity_gap_s": 10.0},
            "observations": [
                {"t": 0.0, "subject": "PO-4471"},
                {"t": 4.0, "subject": "PO-4471"},
                {"t": 9.5, "subject": "PO-4471"},
            ],
            "expect": {"episodes": 1, "reasons": ["new_subject", "continued", "continued"]},
        },
        {
            "name": "after_gap_opens_new",
            "policy": {"continuity_gap_s": 10.0},
            "observations": [
                {"t": 0.0, "subject": "PO-4471"},
                {"t": 25.0, "subject": "PO-4471"},
            ],
            "expect": {"episodes": 2, "reasons": ["new_subject", "gap_exceeded"]},
        },
        {
            "name": "aba_no_spurious_third",
            "policy": {"continuity_gap_s": 10.0},
            "observations": [
                {"t": 0.0, "subject": "A"},
                {"t": 1.0, "subject": "B"},
                {"t": 2.0, "subject": "A"},
            ],
            "expect": {"episodes": 2, "reasons": ["new_subject", "new_subject", "continued"]},
        },
        {
            "name": "terminal_reopen_disallowed",
            "policy": {"continuity_gap_s": 600.0, "allow_reopen": False},
            "close_after": 0,
            "observations": [
                {"t": 0.0, "subject": "PO-4471"},
                {"t": 30.0, "subject": "PO-4471"},
            ],
            "expect": {"episodes": 2, "reasons": ["new_subject", "episode_terminal"]},
        },
        {
            "name": "terminal_reopen_allowed",
            "policy": {"continuity_gap_s": 1800.0, "allow_reopen": True},
            "close_after": 0,
            "observations": [
                {"t": 0.0, "subject": "claim-1"},
                {"t": 30.0, "subject": "claim-1"},
            ],
            "expect": {"episodes": 1, "reasons": ["new_subject", "reopened"]},
        },
    ]
}


def _load_cases() -> list[dict]:
    if _FIXTURE.exists():
        return json.loads(_FIXTURE.read_text())["cases"]
    return _EMBEDDED["cases"]


class TestContinuationStreams(unittest.TestCase):
    """The four review cases, replayed observation by observation through the typed decision."""

    def test_stream_cases(self):
        for case in _load_cases():
            with self.subTest(case=case["name"]):
                policy = EpisodePolicy(
                    continuity_gap_s=case["policy"].get("continuity_gap_s"),
                    allow_reopen=bool(case["policy"].get("allow_reopen", False)),
                )
                log = EpisodeLog("g", policy=policy)
                close_after = case.get("close_after")
                reasons = []
                for i, obs in enumerate(case["observations"]):
                    d = log.observe_decision(obs["t"], obs["subject"], subject_key=obs["subject"])
                    reasons.append(d.reason)
                    if close_after is not None and i == close_after:
                        log.close(d.episode.id)
                self.assertEqual(reasons, case["expect"]["reasons"])
                self.assertEqual(log.count(), case["expect"]["episodes"])

    def test_embedded_cases_match_the_fixture_when_present(self):
        # The embedded copy is a fallback, not a second source of truth — if the synthetic tree is
        # there, the two must agree or one of them is silently drifting.
        if not _FIXTURE.exists():
            self.skipTest("eval/synthetic/continuation not present in this worktree")
        on_disk = {c["name"]: c["expect"] for c in json.loads(_FIXTURE.read_text())["cases"]}
        embedded = {c["name"]: c["expect"] for c in _EMBEDDED["cases"]}
        self.assertEqual(on_disk, embedded)


class TestThreeDecisionsAreSeparate(unittest.TestCase):
    def test_same_subject_is_evidence_not_definition(self):
        # THE card. Exact subject equality holds in both observations; only the continuation
        # conjunction differs, and it alone decides.
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=10.0))
        first = log.observe_decision(0.0, "PO-4471", subject_key="PO-4471")
        late = log.observe_decision(100.0, "PO-4471", subject_key="PO-4471")
        self.assertTrue(late.subject_matched)          # decision (1): same subject
        self.assertTrue(late.is_new)                   # decision (2): NOT the same episode
        self.assertEqual(late.reason, "gap_exceeded")
        self.assertEqual(late.prior_episode_id, first.episode.id)
        self.assertEqual(late.gap_s, 100.0)
        self.assertTrue(late.denied)

    def test_mutation_seam_can_refuse_continuation(self):
        # Decision (3)'s seam (the implementation owns the channel). A finding that contradicts the episode
        # must open a new one rather than silently mutating the old one.
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=10.0))
        log.observe_decision(0.0, {"total": 100}, subject_key="INV-1")
        d = log.observe_decision(
            1.0, {"total": 250}, subject_key="INV-1",
            mutation=lambda ep, obs: ep.exemplar["total"] == obs["total"],
        )
        self.assertTrue(d.subject_matched)
        self.assertTrue(d.is_new)
        self.assertEqual(d.reason, "mutation_incompatible")

    def test_mutation_default_is_compatible(self):
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=10.0))
        log.observe_decision(0.0, "x", subject_key="s")
        self.assertEqual(log.observe_decision(1.0, "x", subject_key="s").reason, "continued")

    def test_status_conjunct_is_independent_of_the_gap(self):
        # Inside the gap, same subject — but the episode is finished. Terminal wins.
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1000.0))
        d1 = log.observe_decision(0.0, "po", subject_key="po")
        log.close(d1.episode.id)
        d2 = log.observe_decision(1.0, "po", subject_key="po")
        self.assertEqual(d2.reason, "episode_terminal")
        self.assertNotEqual(d2.episode.id, d1.episode.id)

    def test_reopen_returns_the_same_episode_to_open(self):
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1000.0, allow_reopen=True))
        d1 = log.observe_decision(0.0, "po", subject_key="po")
        log.close(d1.episode.id)
        d2 = log.observe_decision(1.0, "po", subject_key="po")
        self.assertEqual(d2.reason, "reopened")
        self.assertEqual(d2.episode.id, d1.episode.id)
        self.assertEqual(d2.episode.status, "open")
        self.assertEqual(log.count(), 1)

    def test_reasons_are_the_closed_vocabulary(self):
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1.0))
        seen = {log.observe_decision(float(t), "s", subject_key="s").reason for t in (0, 10, 11)}
        self.assertTrue(seen <= set(CONTINUATION_REASONS))

    def test_subject_key_short_circuits_the_judge(self):
        # The Y5/Z3 seam: with a judge-emitted subject key the judge is not consulted at all.
        calls = []

        def judge(summaries, exemplar):
            calls.append(exemplar)
            return summaries[-1]["id"]

        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=10.0))
        log.observe_decision(0.0, "a", judge, subject_key="A")
        d = log.observe_decision(1.0, "b", judge, subject_key="B")
        self.assertEqual(calls, [])
        self.assertEqual(d.reason, "new_subject")

    def test_judge_match_still_passes_the_continuation_conjunction(self):
        # The judge says "same subject as ep1" — and is OVERRULED on continuation by the gap.
        judge = lambda s, e: s[-1]["id"]  # noqa: E731
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=5.0))
        log.observe(0.0, "cup", judge)
        _, is_new = log.observe(100.0, "cup", judge)
        self.assertTrue(is_new)
        self.assertEqual(log.count(), 2)

    def test_representative_is_live_on_continuation(self):
        # Was a dead field before Y1: set once at open and never updated.
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=10.0))
        log.observe_decision(0.0, "frame-1", subject_key="s")
        d = log.observe_decision(1.0, "frame-2", subject_key="s")
        self.assertEqual(d.episode.exemplar, "frame-1")       # the opening observation
        self.assertEqual(d.episode.representative, "frame-2")  # ...and the current one
        self.assertEqual(d.episode.observations, 2)

    def test_denied_continuation_is_never_silent(self):
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1.0))
        log.observe_decision(0.0, "x", subject_key="s")
        with self.assertLogs("percept.episodes", level="INFO") as cm:
            log.observe_decision(50.0, "x", subject_key="s")
        self.assertTrue(any("episode.continuation_denied" in line for line in cm.output))

    def test_on_continuation_hook_sees_every_decision(self):
        seen = []
        log = EpisodeLog(
            "g", policy=EpisodePolicy(continuity_gap_s=5.0), on_continuation=seen.append
        )
        log.observe_decision(0.0, "x", subject_key="s")
        log.observe_decision(1.0, "x", subject_key="s")
        log.observe_decision(50.0, "x", subject_key="s")
        self.assertEqual([d.reason for d in seen], ["new_subject", "continued", "gap_exceeded"])


class TestEpisodePolicyIsPerConcern(unittest.TestCase):
    def test_default_policy_is_the_legacy_unbounded_one(self):
        # Byte-for-byte: an un-authored concern must not suddenly split its episodes.
        p = EpisodePolicy()
        self.assertIsNone(p.continuity_gap_s)
        self.assertFalse(p.allow_reopen)
        log = EpisodeLog("g")
        log.observe_decision(0.0, "x", subject_key="s")
        self.assertFalse(log.observe_decision(10_000.0, "x", subject_key="s").is_new)

    def test_shipped_presets_do_not_share_semantics(self):
        gaps = {name: p.continuity_gap_s for name, p in EPISODE_POLICIES.items()}
        self.assertNotEqual(gaps["document_presence"], gaps["door_interval"])
        self.assertTrue(EPISODE_POLICIES["conversational_claim"].allow_reopen)
        self.assertIsNone(EPISODE_POLICIES["unbounded"].continuity_gap_s)

    def test_policy_of_goal_reads_authored_state(self):
        goal = Goal(id="g", condition="c", say="s", state={
            "episode": {"policy": "door_interval", "allow_reopen": True}
        })
        p = episode_policy_of(goal)
        self.assertEqual(p.continuity_gap_s, 5.0)  # from the named preset
        self.assertTrue(p.allow_reopen)            # ...with the explicit override on top

    def test_policy_of_goal_without_authoring_is_the_default(self):
        goal = Goal(id="g", condition="c", say="s", state={})
        self.assertEqual(episode_policy_of(goal), EpisodePolicy())

    def test_terminal_statuses_are_authorable(self):
        goal = Goal(id="g", condition="c", say="s", state={
            "episode": {"terminal_statuses": ["settled", "void"]}
        })
        p = episode_policy_of(goal)
        self.assertTrue(p.is_terminal("void"))
        self.assertFalse(p.is_terminal(CLOSED))  # replaced, not merged — the concern's vocabulary

    def test_coercer_rejects_malformed_authoring(self):
        self.assertEqual(coerce_authored_episode("nope"), {})
        self.assertEqual(coerce_authored_episode({"policy": "made_up"}), {})
        self.assertEqual(coerce_authored_episode({"continuity_gap_s": -5}), {})
        self.assertEqual(coerce_authored_episode({"continuity_gap_s": True}), {})
        self.assertEqual(coerce_authored_episode({"terminal_statuses": [1, 2]}), {})

    def test_coercer_clamps_and_passes_valid_authoring(self):
        out = coerce_authored_episode({
            "policy": "document_presence", "continuity_gap_s": 10**9,
            "allow_reopen": True, "terminal_statuses": ["done", "done", "x" * 100],
        })["episode"]
        self.assertEqual(out["policy"], "document_presence")
        self.assertEqual(out["continuity_gap_s"], 86_400.0)  # clamped to a day
        self.assertTrue(out["allow_reopen"])
        self.assertEqual(out["terminal_statuses"], ["done"])  # deduped; the oversized one dropped

    def test_coerced_authoring_round_trips_into_a_policy(self):
        state = coerce_authored_episode({"policy": "door_interval"})
        goal = Goal(id="g", condition="c", say="s", state=dict(state))
        self.assertEqual(episode_policy_of(goal), EPISODE_POLICIES["door_interval"])


class _SpyGate(Gate):
    """gate.py stays MECHANICALLY UNCHANGED — episode state reaches it only as a rearm() call."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rearms: list = []

    def rearm(self, key):
        self.rearms.append(key)
        super().rearm(key)


class TestExecutorSeam(unittest.TestCase):
    def _goal(self, **state):
        return Goal(id="g", condition="c", say="fired", state={"type": "count", **state})

    def test_aba_revisit_does_not_fire_a_third_time(self):
        # A-B-A: the third observation CONTINUES A. It never reaches the gate as a new episode, so
        # no third fire and no third re-arm.
        gate = _SpyGate()
        ex = BehaviorExecutor(gate)
        goal, prog = self._goal(), BehaviorProgram(sense="episode")
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=10.0))
        fires = []
        for t, subject in ((0.0, "A"), (1.0, "B"), (2.0, "A")):
            d = log.observe_decision(t, subject, subject_key=subject)
            fired = ex.fire_episode(
                goal, prog, n=log.count(), is_new=d.is_new, t=t, episode_id=d.episode.id
            )
            if fired is not None:
                fires.append((t, fired.entity_id))
        self.assertEqual(fires, [(0.0, 1), (1.0, 2)])
        self.assertEqual(len(gate.rearms), 1)  # only B rolled the episode identity

    def test_withheld_b_leaves_no_edge_for_a_to_ride(self):
        # "B fails its bar, A must not re-fire." A withheld new-episode fire (here the accumulator
        # bar; a dwell is the same class of withholding) must not re-arm the shared episode key, and
        # A's later revisit is a CONTINUATION, so nothing fires on it either.
        gate = _SpyGate()
        ex = BehaviorExecutor(gate)
        goal = self._goal()
        prog = BehaviorProgram(sense="episode", accumulate=Accumulator(fire_when="n != 2"))
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=10.0))
        fires = []
        for t, subject in ((0.0, "A"), (1.0, "B"), (2.0, "A")):
            d = log.observe_decision(t, subject, subject_key=subject)
            fired = ex.fire_episode(
                goal, prog, n=log.count(), is_new=d.is_new, t=t, episode_id=d.episode.id
            )
            if fired is not None:
                fires.append(t)
        self.assertEqual(fires, [0.0])   # A fired once; B withheld; A's revisit silent
        self.assertEqual(gate.rearms, [])  # a withheld fire never re-arms the singleton key

    def test_a_genuine_second_occurrence_still_fires(self):
        # The counterfactual that keeps this honest: separation must not SUPPRESS real repeats. The
        # same subject after the gap is a new episode and DOES fire.
        gate = _SpyGate()
        ex = BehaviorExecutor(gate)
        goal, prog = self._goal(), BehaviorProgram(sense="episode")
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=10.0))
        fires = []
        for t in (0.0, 100.0):
            d = log.observe_decision(t, "A", subject_key="A")
            fired = ex.fire_episode(
                goal, prog, n=log.count(), is_new=d.is_new, t=t, episode_id=d.episode.id
            )
            if fired is not None:
                fires.append(t)
        self.assertEqual(fires, [0.0, 100.0])
        self.assertEqual(gate.rearms, [("g", "__episode__")])

    def test_episode_id_absent_keeps_the_legacy_n_discriminator(self):
        gate = _SpyGate()
        ex = BehaviorExecutor(gate)
        goal, prog = self._goal(), BehaviorProgram(sense="episode")
        self.assertIsNotNone(ex.fire_episode(goal, prog, n=1, is_new=True))
        self.assertIsNotNone(ex.fire_episode(goal, prog, n=2, is_new=True))
        self.assertEqual(gate.rearms, [("g", "__episode__")])

    def test_gate_never_learns_about_episodes(self):
        # The gate must stay a pure key -> armed bit: no episode/policy/subject state on it.
        gate = Gate()
        for forbidden in ("policy", "episodes", "subject", "continuity_gap_s"):
            self.assertFalse(hasattr(gate, forbidden))


class TestPerceptCountEventDecision(unittest.TestCase):
    def _agent(self, **episode_state):
        a = Percept.create(discover_plugins=False)
        state = {"type": "count", "count": 0}
        if episode_state:
            state["episode"] = episode_state
        a.add_goal(Goal(id="cups", condition="a cup", say="another cup", state=state))
        return a

    def test_agent_uses_the_concerns_policy(self):
        a = self._agent(continuity_gap_s=10.0)
        _, new1, d1 = a.count_event_decision("cups", "cup", t=1.0, subject_key="cup")
        _, new2, d2 = a.count_event_decision("cups", "cup", t=5.0, subject_key="cup")
        _, new3, d3 = a.count_event_decision("cups", "cup", t=500.0, subject_key="cup")
        self.assertEqual([new1, new2, new3], [True, False, True])
        self.assertEqual([d1.reason, d2.reason, d3.reason],
                         ["new_subject", "continued", "gap_exceeded"])
        self.assertEqual(a.goals.get("cups").state["count"], 2)
        self.assertEqual(a.metrics.get("episode.denied.gap_exceeded"), 1)

    def test_count_event_signature_is_backwards_compatible(self):
        a = self._agent()
        always_continue = lambda s, e: s[-1]["id"] if s else None  # noqa: E731
        ep, is_new = a.count_event("cups", "cupA", always_continue, t=1.0)
        self.assertTrue(is_new)
        self.assertFalse(a.count_event("cups", "cupA", always_continue, t=2.0)[1])
        self.assertEqual(a.goals.get("cups").state["count"], 1)
        self.assertEqual(ep.goal_id, "cups")


if __name__ == "__main__":
    unittest.main()
