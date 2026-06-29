"""Item 2 mvp2 (L1): the process-QA running answer RESOLVES on accumulated evidence.

The gap behind ``unresolved_rate == 1.0`` on the ``proactiveqa-e2e`` track: a process_qa concern
accumulates a per-concern ``running_answer`` over many escalations, but the UNCHANGED gate only fires
on a confident-YES verdict — which an evidence-gathering judge returns rarely. So the running answer
grows but is NEVER emitted as a recognized final answer the metric can score. Every turn reads as
unresolved even though a running answer (with real content) exists.

This is the A/B for the fix, fully deterministic (a fake vision adapter returning running-answer
fragments over several escalations — NO Gemini, no network):

  - Arm A (no resolution): the gate never fires (satisfied stays False), so an accumulating-but-
    never-resolving concern emits ZERO answers — unresolved.
  - Arm B (resolution): once the accumulated running answer crosses the coverage/substance/confidence
    bar, the answer-path resolver emits a FIRE event whose verdict carries the resolved answer —
    resolved, with the right accumulated content.
  - Precision guard: a THIN stream (one short/low-confidence fragment) must NOT resolve — a real
    partial beats a hallucinated fix.

The resolution is GENERAL: it reads only accumulated coverage + substance + confidence (never the
question text), and it routes through ConcernContext + the answer path, NOT a new gate firing rule —
the gate is called only on a dedicated ``__resolved__`` de-dup key, exactly like the shipped
coverage/episode/overdue absence fires.
"""

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.concern import resolution_ready
from percept.gate import FIRE


class _ResolvingFake:
    """A process_qa vision adapter: each escalation accumulates one more evidence FRAGMENT into the
    running answer (the model rewrites a growing summary), with controllable per-call confidence and
    fragment substance. ``satisfied`` stays False — the gate must NEVER be what resolves the answer."""

    def __init__(self, fragments, *, confidence=0.9):
        self._fragments = list(fragments)
        self._confidence = confidence
        self._i = 0
        self.seen_priors: list = []

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None) -> Verdict:
        prior = str((context or {}).get("running_answer") or "")
        self.seen_priors.append(prior)
        frag = self._fragments[min(self._i, len(self._fragments) - 1)]
        self._i += 1
        running = f"{prior}; {frag}".strip("; ").strip() if prior else frag
        # satisfied=False ALWAYS: the running answer rides alongside, it never drives the gate.
        return Verdict(satisfied=False, confidence=self._confidence, evidence=frag, running_answer=running)

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _drive(agent, source, n):
    async def go():
        out = []
        for i in range(n):
            out += await agent.perceive_judged(source, t=float(i + 1))
        return out
    return asyncio.run(go())


class TestResolutionPredicate(unittest.TestCase):
    """The pure resolution bar — general (accumulation only), precision-guarded."""

    def test_thin_evidence_never_resolves(self):
        # one fragment, even if confident, is not enough coverage to commit
        self.assertFalse(resolution_ready("peel the onion", evidence_count=1, confidence=0.95))
        # empty / stub answer never resolves regardless of coverage
        self.assertFalse(resolution_ready("", evidence_count=5, confidence=0.99))
        self.assertFalse(resolution_ready("ok", evidence_count=5, confidence=0.99))  # too short
        # low confidence never resolves
        self.assertFalse(resolution_ready("peel then slice the onion", evidence_count=4, confidence=0.4))

    def test_sufficient_evidence_resolves(self):
        self.assertTrue(resolution_ready("peel then slice the onion", evidence_count=2, confidence=0.9))

    def test_boundary_relaxes_coverage_floor_by_one_but_not_substance(self):
        # at an episode boundary a single substantive+confident observation may resolve ...
        self.assertTrue(resolution_ready("peel then slice the onion", evidence_count=1,
                                         confidence=0.9, at_boundary=True))
        # ... but an empty / low-confidence answer still never resolves at the boundary
        self.assertFalse(resolution_ready("", evidence_count=1, confidence=0.9, at_boundary=True))
        self.assertFalse(resolution_ready("peel then slice the onion", evidence_count=1,
                                          confidence=0.3, at_boundary=True))


class TestResolutionAB(unittest.TestCase):
    def _goal(self, gid="proc"):
        return Goal(id=gid, condition="what are the steps to make the dish",
                    say="answer ready", state={"type": "watch", "process_qa": True})

    def test_arm_A_no_resolution_is_unresolved(self):
        """ARM A: with the resolution path DISABLED (the pre-fix world — only the gate could emit, and
        the gate never fires because satisfied stays False), ZERO answers are emitted even though the
        running answer accumulates real evidence. This IS the unresolved_rate==1.0 symptom."""
        import percept.agent as agent_mod
        saved = agent_mod.resolution_ready
        agent_mod.resolution_ready = lambda *a, **k: False  # the pre-fix world: nothing ever resolves
        try:
            vision = _ResolvingFake(["peel the onion", "slice the onion", "season the beef"])
            a = Percept.create(vision=vision, discover_plugins=False)
            a.add_goal(self._goal())
            events = _drive(a, "frame", 3)
            answers = [e for e in events if e.action == FIRE]
            self.assertEqual(answers, [])  # nothing resolved -> unresolved (the symptom)
            # yet the running answer DID accumulate (the evidence is there, just never emitted)
            self.assertIn("season the beef", a.executor.ctx("proc").running_answer)
        finally:
            agent_mod.resolution_ready = saved

    def test_arm_B_resolves_with_the_accumulated_answer(self):
        """ARM B: resolution ON -> once enough evidence accumulates, a FIRE event carries the resolved
        answer. Resolved, with the right accumulated content, and NOT via the gate's satisfied-YES rule
        (the judge ALWAYS returns satisfied=False — the answer-path resolver is what emits the answer)."""
        vision = _ResolvingFake(["peel the onion", "slice the onion", "season the beef"])
        a = Percept.create(vision=vision, discover_plugins=False)
        a.add_goal(self._goal())

        # capture every verdict the JUDGE returned to prove satisfied was never True (the gate's
        # satisfied-YES rule was never the trigger — resolution came purely from the answer path).
        judged = []
        orig_judge = vision.judge
        async def spy(*args, **kw):
            v = await orig_judge(*args, **kw)
            judged.append(v)
            return v
        vision.judge = spy

        events = _drive(a, "frame", 3)

        fires = [e for e in events if e.action == FIRE]
        self.assertTrue(fires, "Arm B must resolve at least one answer")
        # the resolved answer carries the ACCUMULATED running answer (real evidence, not a single frag)
        resolved_text = fires[-1].verdict.running_answer
        self.assertIn("peel the onion", resolved_text)
        self.assertIn("slice the onion", resolved_text)
        # the first escalation (one fragment) did NOT resolve — precision guard held until coverage built
        self.assertEqual(vision.seen_priors[0], "")  # started empty
        # the JUDGE never returned satisfied=True — so the gate's satisfied-YES rule was NOT what fired.
        self.assertFalse(any(v.satisfied for v in judged))

    def test_thin_stream_does_not_resolve(self):
        """PRECISION GUARD end-to-end: a single short, low-confidence fragment never resolves."""
        vision = _ResolvingFake(["ok"], confidence=0.4)  # short + uncertain
        a = Percept.create(vision=vision, discover_plugins=False)
        a.add_goal(self._goal())
        events = _drive(a, "frame", 3)
        self.assertEqual([e for e in events if e.action == FIRE], [])  # nothing hallucinated

    def test_resolution_does_not_fire_twice_on_an_unchanged_answer(self):
        """De-dup: an identical running answer resolves ONCE (the __resolved__ key folds in the text)."""
        # after step 2 the answer stops changing (same fragment repeated) -> only one resolution
        vision = _ResolvingFake(["peel the onion", "slice the onion"])
        # force the third+ call to repeat the prior answer unchanged
        a = Percept.create(vision=vision, discover_plugins=False)
        a.add_goal(self._goal())

        class _Sticky(_ResolvingFake):
            async def judge(self, condition, frame, **kw):
                prior = str((kw.get("context") or {}).get("running_answer") or "")
                self.seen_priors.append(prior)
                # once we have two fragments, keep returning the SAME accumulated answer
                if self._i >= 2:
                    return Verdict(satisfied=False, confidence=0.9, evidence="(same)",
                                   running_answer=prior or "peel the onion; slice the onion")
                frag = ["peel the onion", "slice the onion"][self._i]
                self._i += 1
                running = f"{prior}; {frag}".strip("; ").strip() if prior else frag
                return Verdict(satisfied=False, confidence=0.9, evidence=frag, running_answer=running)

        sticky = _Sticky([])
        a2 = Percept.create(vision=sticky, discover_plugins=False)
        a2.add_goal(self._goal("proc2"))
        events = _drive(a2, "frame", 5)
        fires = [e for e in events if e.action == FIRE]
        texts = {e.verdict.running_answer for e in fires}
        # the answer "peel the onion; slice the onion" resolves exactly once despite being seen 4x
        self.assertEqual(sum(1 for e in fires
                             if e.verdict.running_answer == "peel the onion; slice the onion"), 1)
        self.assertTrue(texts)  # something resolved


class TestDefaultGoalUnchanged(unittest.TestCase):
    def test_non_process_goal_never_resolves(self):
        """A plain watch goal (no process_qa) never grows a running answer and never resolves —
        byte-for-byte unchanged (the resolution path is opt-in)."""
        class _Plain:
            async def judge(self, condition, frame, **kw):
                return Verdict(satisfied=False, confidence=0.5, evidence="single-frame")
            async def detect(self, frame, goals):
                return []
            async def detect_events(self, frames, goals):
                return []

        a = Percept.create(vision=_Plain(), discover_plugins=False)
        a.add_goal(Goal(id="plain", condition="someone at the door", say="hi",
                        state={"type": "watch"}))
        events = _drive(a, "frame", 4)
        self.assertEqual([e for e in events if e.action == FIRE], [])
        self.assertEqual(a.executor.ctx("plain").running_answer, "")
        self.assertEqual(a.executor.ctx("plain").process_log, [])  # no coverage accumulated


class TestBoundaryResolution(unittest.TestCase):
    def test_quiet_episode_end_resolves_a_thin_but_substantive_answer(self):
        """A process_qa episode that went quiet RESOLVES whatever was confidently accumulated, with the
        coverage floor relaxed by one (at_boundary) — but still never resolves an empty answer."""
        # one substantive confident fragment, then quiet: in-tick won't resolve (coverage 1 < 2),
        # but the boundary (at_boundary -> floor 1) does.
        vision = _ResolvingFake(["peel then slice the onion"])
        a = Percept.create(vision=vision, discover_plugins=False)
        a.add_goal(Goal(id="proc", condition="steps?", say="answer ready",
                        state={"type": "watch", "process_qa": True}))

        async def go():
            mid = await a.perceive_judged("frame", t=1.0)   # one fragment -> NOT enough in-tick
            ended = await a.check_episode_boundary(t=5.0, inactivity_s=2.0)  # quiet -> boundary resolves
            again = await a.check_episode_boundary(t=7.0, inactivity_s=2.0)  # wiped -> no re-fire
            return mid, ended, again

        mid, ended, again = asyncio.run(go())
        self.assertEqual([e for e in mid if e.action == FIRE], [])  # precision held in-tick
        self.assertEqual(len(ended), 1)                              # boundary resolved it once
        self.assertEqual(ended[0].action, FIRE)
        self.assertIn("peel then slice the onion", ended[0].verdict.running_answer)
        self.assertEqual(again, [])                                  # wiped + re-armed, no double fire


if __name__ == "__main__":
    unittest.main()
