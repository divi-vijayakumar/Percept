"""Episode/entity-memory judge mode — a per-goal running answer threaded across
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
    EPISODE_CONTEXT_MAX_CHARS,
    JUDGE_MODE_EPISODE_MEMORY,
    EpisodeMemoryState,
    EpisodeMemoryStore,
    episode_judge_context,
    episode_memory_enabled,
    judge_episode_memory,
    with_open_episode,
)
from percept.episodes import JUDGE_CONTEXT_MAX_REVISIONS, EpisodeLog
from percept.judge_context import JUDGE_OPEN_EPISODE_KEY, ReservedContextKeyError


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


class OpenEpisodeSerializationTests(unittest.TestCase):
    """The OTHER half of the memory: what the SYSTEM established, serialized for the judge.

    The running answer above carries what the model said; these carry the concern's open episode, so
    a persistent question ("is it still there?") is answered as a continuation instead of being
    re-decided from one frame with no memory."""

    @staticmethod
    def _log_with_open_episode(*, confirmations: int = 0, value: str | None = None) -> EpisodeLog:
        log = EpisodeLog("watch")
        episode, _ = log.observe(10.0, "exemplar-payload", None, subject_key="backpack_a")
        if value is not None:
            log.annotate(episode, value=value)
        for i in range(confirmations):
            log.record_revision(episode, "confirmed", t=11.0 + i)
        return log

    def test_no_log_and_empty_log_produce_no_context(self):
        self.assertIsNone(episode_judge_context(None))
        self.assertIsNone(episode_judge_context(EpisodeLog("watch")))

    def test_closed_episode_produces_no_context(self):
        log = self._log_with_open_episode()
        log.close(log.current_open().id, t=50.0)
        self.assertIsNone(episode_judge_context(log))

    def test_open_episode_is_serialized_with_its_identity_and_history(self):
        log = self._log_with_open_episode(confirmations=3, value="an unattended black backpack")
        text = episode_judge_context(log)

        self.assertIsNotNone(text)
        self.assertIn("backpack_a", text)                 # the subject it is about
        self.assertIn("an unattended black backpack", text)  # what was already established
        self.assertIn("t=10.000", text)                   # when it started
        self.assertIn("open", text)                       # its lifecycle state

    def test_serialized_episode_carries_no_payload(self):
        """Pixel-free: the caller's opaque observation payload must never reach a prompt."""
        log = self._log_with_open_episode(confirmations=2)
        episode = log.current_open()
        self.assertEqual(episode.exemplar, "exemplar-payload")  # it IS on the episode
        self.assertNotIn("exemplar-payload", episode_judge_context(log))  # and NOT in the context

        projected = episode.to_judge_dict()
        self.assertNotIn("exemplar", projected)
        self.assertNotIn("representative", projected)

    def test_size_is_bounded_under_many_confirmations(self):
        """The endurance property: a 24/7 stream must not grow the prompt.

        A confirmation coalesces into scalars rather than appending a row, so an episode confirmed
        4,000 times must serialize to essentially the same size as one confirmed twice — never 2,000x
        the size."""
        short = episode_judge_context(self._log_with_open_episode(confirmations=2, value="a bag"))
        long = episode_judge_context(self._log_with_open_episode(confirmations=4000, value="a bag"))

        self.assertLessEqual(len(long), len(short) + 32)  # only the digits of the clock may grow
        self.assertLessEqual(len(long), EPISODE_CONTEXT_MAX_CHARS)

    def test_size_is_bounded_under_many_state_changes(self):
        """The timeline is truncated to its most recent rows, and the omission is DECLARED."""
        log = self._log_with_open_episode()
        episode = log.current_open()
        for i in range(400):
            log.record_revision(episode, "updated", t=20.0 + i, value=f"finding {i}")

        text = episode_judge_context(log)
        projected = episode.to_judge_dict()

        self.assertLessEqual(len(projected["revisions"]), JUDGE_CONTEXT_MAX_REVISIONS)
        self.assertGreater(projected["revisions_omitted"], 0)
        self.assertLessEqual(len(text), EPISODE_CONTEXT_MAX_CHARS)
        self.assertIn("finding 399", text)      # the most RECENT changes are the ones kept
        self.assertNotIn("finding 0,", text)    # the oldest are what gets dropped first

    def test_size_is_bounded_under_a_pathological_value(self):
        log = self._log_with_open_episode(value="X" * 100_000)
        self.assertLessEqual(len(episode_judge_context(log)), EPISODE_CONTEXT_MAX_CHARS)

    def test_many_open_episodes_contribute_one_plus_a_count(self):
        """Size must not grow with the number of open episodes either."""
        log = EpisodeLog("watch")
        for i in range(40):
            log.observe(10.0 + i, f"obs-{i}", None, subject_key=f"subject_{i}")

        text = episode_judge_context(log)
        self.assertEqual(len(log.open_episodes()), 40)
        self.assertIn("subject_39", text)      # the most recently seen one is the one shown
        self.assertNotIn("subject_0,", text)
        self.assertIn("39 other open episode(s) not shown", text)
        self.assertLessEqual(len(text), EPISODE_CONTEXT_MAX_CHARS)

    def test_injection_copies_the_caller_context_and_refuses_to_overwrite(self):
        caller = {"process_qa": True}
        merged = with_open_episode(caller, "episode text")

        self.assertEqual(merged[JUDGE_OPEN_EPISODE_KEY], "episode text")
        self.assertNotIn(JUDGE_OPEN_EPISODE_KEY, caller)  # caller's dict never mutated
        with self.assertRaises(ReservedContextKeyError):
            with_open_episode(merged, "second text")

    def test_running_answer_path_can_also_carry_the_open_episode(self):
        vision = _StepFake()
        goal = Goal(id="g", condition="is the bag still there?", say="x",
                    state={"judge_mode": JUDGE_MODE_EPISODE_MEMORY})
        store = EpisodeMemoryStore()
        log = self._log_with_open_episode(confirmations=2, value="a black backpack")

        async def go():
            await judge_episode_memory(vision, goal, "frame", store=store, episode_log=log)
            await judge_episode_memory(vision, goal, "frame", store=store)  # no log -> unchanged

        asyncio.run(go())

        with_log, without_log = vision.seen_contexts
        self.assertIn("a black backpack", with_log[JUDGE_OPEN_EPISODE_KEY])
        self.assertTrue(with_log["process_qa"])            # both halves ride together
        self.assertNotIn(JUDGE_OPEN_EPISODE_KEY, without_log)


if __name__ == "__main__":
    unittest.main()
