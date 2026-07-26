"""the implementation (E3): the episode/entity-memory judge mode — a per-goal running answer threaded across
successive judged frames, integrating evidence over time (the next lever after the ordered window).

All offline against a fake ``Vision`` adapter — wiring, not intelligence (no live model needed in the
test env). Mirrors the seam the production backends already implement
(``backends/openrouter.py:_judge_process_qa``): ``context={"process_qa": True, "running_answer": ...}``
in, ``Verdict.running_answer`` out. This module is additive — it does not touch ``agent.py``'s
existing single-frame/window judge paths, so those are exercised elsewhere (test_process_qa.py etc.)
and are out of scope here.
"""

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.episode_memory import (
    JUDGE_MODE_EPISODE_MEMORY,
    EpisodeMemoryState,
    EpisodeMemoryStore,
    episode_memory_enabled,
    judge_episode_memory,
)


class _StepFake:
    """Records the prior running answer it was handed and returns prior+stepN (no model). Never
    reports its own ``satisfied`` bit true — so any integrated satisfied verdict the mode returns must
    come from ACCUMULATED evidence, not the raw per-frame model read."""

    def __init__(self, *, confidence: float = 0.9) -> None:
        self.confidence = confidence
        self.seen_contexts: list = []
        self._step = 0

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None) -> Verdict:
        self.seen_contexts.append(context)
        prior = str((context or {}).get("running_answer") or "")
        self._step += 1
        summary = f"{prior} step{self._step}".strip()
        return Verdict(satisfied=False, confidence=self.confidence, evidence=summary,
                       running_answer=summary)


class _BlankFake:
    """A backend that fails to parse (returns no running_answer) — the "parse miss" case."""

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None) -> Verdict:
        return Verdict(satisfied=False, confidence=0.1, evidence="")


def _drive(vision, goal, store, n):
    async def go():
        out = []
        for _ in range(n):
            out.append(await judge_episode_memory(vision, goal, "frame", store=store))
        return out
    return asyncio.run(go())


class TestEpisodeMemoryFlag(unittest.TestCase):
    def test_disabled_by_default(self):
        goal = Goal(id="g", condition="x", say="x", state={"type": "watch"})
        self.assertFalse(episode_memory_enabled(goal))

    def test_enabled_via_judge_mode_flag(self):
        goal = Goal(id="g", condition="x", say="x",
                    state={"type": "watch", "judge_mode": JUDGE_MODE_EPISODE_MEMORY})
        self.assertTrue(episode_memory_enabled(goal))

    def test_other_judge_mode_value_is_not_enabled(self):
        goal = Goal(id="g", condition="x", say="x", state={"type": "watch", "judge_mode": "window"})
        self.assertFalse(episode_memory_enabled(goal))


class TestRunningAnswerAccumulates(unittest.TestCase):
    def test_running_answer_threads_in_and_accumulates_out(self):
        vision = _StepFake()
        store = EpisodeMemoryStore()
        goal = Goal(id="proc", condition="what steps has the process gone through?", say="answer ready",
                    state={"type": "watch", "judge_mode": JUDGE_MODE_EPISODE_MEMORY})

        verdicts = _drive(vision, goal, store, 3)

        # the PRIOR running answer threaded INTO each judge call (starts empty, then accumulates)
        priors = [c.get("running_answer") for c in vision.seen_contexts]
        self.assertEqual(priors, ["", "step1", "step1 step2"])
        # the model's rewritten summary accumulated OUT onto the per-goal state
        self.assertEqual(store.get("proc").running_answer, "step1 step2 step3")
        # each returned Verdict carries the running answer at that point in time
        self.assertEqual([v.running_answer for v in verdicts],
                         ["step1", "step1 step2", "step1 step2 step3"])
        # evidence-bearing coverage grew one-per-frame
        self.assertEqual(store.get("proc").frame_count, 3)

    def test_per_goal_state_is_isolated(self):
        store = EpisodeMemoryStore()
        a = Goal(id="a", condition="x", say="x", state={"judge_mode": JUDGE_MODE_EPISODE_MEMORY})
        b = Goal(id="b", condition="y", say="y", state={"judge_mode": JUDGE_MODE_EPISODE_MEMORY})

        _drive(_StepFake(), a, store, 2)
        _drive(_StepFake(), b, store, 1)

        # each goal's accumulated text lives in its own bucket, independent of the other goal's history
        self.assertEqual(store.get("a").running_answer, "step1 step2")
        self.assertEqual(store.get("b").running_answer, "step1")
        self.assertEqual(store.get("a").frame_count, 2)
        self.assertEqual(store.get("b").frame_count, 1)

    def test_parse_miss_preserves_prior_answer(self):
        store = EpisodeMemoryStore()
        goal = Goal(id="g", condition="x", say="x", state={"judge_mode": JUDGE_MODE_EPISODE_MEMORY})
        state = store.get("g")
        state.record("step1")

        verdict = asyncio.run(judge_episode_memory(_BlankFake(), goal, "frame", store=store))

        # a blank/unparsed response never regresses the accumulated answer
        self.assertEqual(verdict.running_answer, "step1")
        self.assertEqual(store.get("g").running_answer, "step1")
        self.assertEqual(store.get("g").frame_count, 1)  # no NEW evidence-bearing frame counted

    def test_reset_clears_state(self):
        state = EpisodeMemoryState()
        state.record("step1")
        state.record("step1 step2")
        self.assertEqual(state.frame_count, 2)

        state.reset()

        self.assertEqual(state.running_answer, "")
        self.assertEqual(state.history, [])
        self.assertEqual(state.frame_count, 0)

    def test_store_bounded_eviction(self):
        store = EpisodeMemoryStore(max_goals=2)
        store.get("a").record("x")
        store.get("b").record("x")
        store.get("c").record("x")  # evicts the oldest ("a")

        self.assertEqual(len(store), 2)
        self.assertNotIn("a", store._states)  # noqa: SLF001 — white-box eviction check
        self.assertIn("c", store._states)  # noqa: SLF001


class TestVerdictReflectsIntegratedEvidence(unittest.TestCase):
    def test_satisfied_integrates_accumulated_coverage_not_a_single_frame(self):
        # The fake NEVER reports satisfied=True on any single frame — so an integrated satisfied=True
        # can only come from accumulated coverage + substance + confidence crossing the resolution bar
        # (percept.concern.resolution_ready), i.e. the mode's own productization on top of the raw
        # per-frame read.
        vision = _StepFake(confidence=0.9)
        store = EpisodeMemoryStore()
        goal = Goal(id="proc", condition="what has happened so far?", say="ready",
                    state={"judge_mode": JUDGE_MODE_EPISODE_MEMORY})

        verdicts = _drive(vision, goal, store, 3)

        # early frames: not enough accumulated coverage/substance yet — below the resolution bar
        self.assertFalse(verdicts[0].satisfied)
        self.assertFalse(verdicts[1].satisfied)
        # by the third frame, accumulated coverage + substance + confidence cross the bar — the
        # verdict is satisfied even though the fake NEVER reports satisfied=True on any single frame
        self.assertTrue(verdicts[2].satisfied)

    def test_low_confidence_never_integrates_to_satisfied(self):
        vision = _StepFake(confidence=0.1)  # below the resolution confidence floor throughout
        store = EpisodeMemoryStore()
        goal = Goal(id="proc", condition="x", say="x", state={"judge_mode": JUDGE_MODE_EPISODE_MEMORY})

        verdicts = _drive(vision, goal, store, 4)

        self.assertTrue(all(not v.satisfied for v in verdicts))

    def test_thin_fragment_never_integrates_to_satisfied(self):
        class _ThinFake:
            async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                            frames=None, audio=None, context=None) -> Verdict:
                return Verdict(satisfied=False, confidence=0.95, evidence="ok", running_answer="ok")

        store = EpisodeMemoryStore()
        goal = Goal(id="proc", condition="x", say="x", state={"judge_mode": JUDGE_MODE_EPISODE_MEMORY})

        verdicts = _drive(_ThinFake(), goal, store, 5)

        # "ok" repeated never carries enough substance (RESOLVE_MIN_CHARS) to integrate to satisfied
        self.assertTrue(all(not v.satisfied for v in verdicts))


if __name__ == "__main__":
    unittest.main()
