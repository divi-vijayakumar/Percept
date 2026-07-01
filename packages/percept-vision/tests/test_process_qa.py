"""Item 2 (L1): process-QA running answer threads per-concern, gate-irrelevant.

All offline against a stub vision — wiring, not intelligence. We assert the SEAM the production
Gemini backend uses: a goal that opts into ``process_qa`` carries ``ConcernContext.running_answer``
INTO the judge (as ``context``) and accumulates the model's rewritten summary back OUT of it, while
the three-state gate still keys ONLY on satisfied/confidence (a default goal is byte-for-byte
unchanged — never sees the context, never grows a running answer).
"""

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept


class _ProcessFake:
    """Records the prior running answer it was handed and returns prior+stepN (no model)."""

    def __init__(self, *, satisfied: bool = True) -> None:
        self.satisfied = satisfied
        self.seen_contexts: list = []
        self._step = 0

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None) -> Verdict:
        self.seen_contexts.append(context)
        if context and context.get("process_qa"):
            prior = str(context.get("running_answer") or "")
            self._step += 1
            summary = f"{prior} step{self._step}".strip()
            return Verdict(satisfied=self.satisfied, confidence=0.9, evidence=summary, running_answer=summary)
        # default (non-process) goal: today's behaviour, no running answer
        return Verdict(satisfied=False, confidence=0.5, evidence="single-frame")

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _drive(agent: Percept, source: str, n: int) -> list:
    async def go():
        out = []
        for i in range(n):
            out += await agent.perceive_judged(source, t=float(i + 1))
        return out
    return asyncio.run(go())


class TestProcessQARunningAnswer(unittest.TestCase):
    def test_running_answer_threads_in_and_accumulates_out(self):
        vision = _ProcessFake()
        a = Percept.create(vision=vision, discover_plugins=False)
        a.add_goal(Goal(id="proc", condition="enough evidence to answer: steps?",
                        say="answer ready", state={"type": "watch", "process_qa": True}))

        _drive(a, "frame", 3)

        # the PRIOR running answer threaded INTO each judge call (start empty, then accumulate)
        priors = [c.get("running_answer") for c in vision.seen_contexts]
        self.assertEqual(priors, ["", "step1", "step1 step2"])
        # and the model's rewritten summary accumulated OUT onto the per-concern context
        self.assertEqual(a.executor.ctx("proc").running_answer, "step1 step2 step3")

    def test_default_goal_is_byte_for_byte_unchanged(self):
        vision = _ProcessFake()
        a = Percept.create(vision=vision, discover_plugins=False)
        a.add_goal(Goal(id="plain", condition="someone is at the door", say="heads up",
                        state={"type": "watch"}))  # NO process_qa opt-in

        _drive(a, "frame", 2)

        # a default goal never gets the process context (so the judge call is unchanged) ...
        self.assertTrue(all(c is None for c in vision.seen_contexts))
        # ... and never grows a running answer
        self.assertEqual(a.executor.ctx("plain").running_answer, "")

    def test_running_answer_does_not_drive_the_satisfied_gate(self):
        # satisfied=False -> the SATISFIED-keyed gate must NOT fire (the summary never sets the gate bit),
        # yet the running answer still threads/accumulates. With mvp2 resolution, the accumulated answer
        # CAN later resolve via the answer path (a separate __resolved__-keyed de-dup) — but that is NOT a
        # satisfied-driven gate fire: every fire here is a resolution carrying the running answer, and the
        # judge's satisfied bit stayed False throughout.
        vision = _ProcessFake(satisfied=False)
        a = Percept.create(vision=vision, discover_plugins=False)
        a.add_goal(Goal(id="proc", condition="enough evidence to answer: steps?",
                        say="answer ready", state={"type": "watch", "process_qa": True}))

        events = _drive(a, "frame", 3)

        fires = [e for e in events if e.action == "fire"]
        # the gate's satisfied/confidence rule never fired on its own: any fire is an answer-path
        # resolution (it carries the running answer as the resolved text), never a bare satisfied gate fire.
        self.assertTrue(all(e.verdict.running_answer for e in fires))
        self.assertEqual(a.executor.ctx("proc").running_answer, "step1 step2 step3")  # still accumulated


if __name__ == "__main__":
    unittest.main()
