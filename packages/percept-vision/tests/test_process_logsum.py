"""Item 4 (L1): log -> summarize -> wipe, triggered by the episode boundary.

All offline against the scripted ``FakeVision`` — wiring, not intelligence. A ``type=='process'`` goal:
  - observes a per-clip step delta each frame and APPENDS it to ``ConcernContext.process_log`` (the VLM
    never sees the log back); it NEVER fires per-clip;
  - ``summarize_process`` consolidates the log on demand (wipe optional);
  - ``check_episode_boundary`` fires the consolidated answer ONCE when no new step has arrived for
    ``inactivity_s`` (sustained absence), wipes the log, and re-arms for the next episode — all through
    the UNCHANGED gate on a dedicated ``__episode__`` key.
"""

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.gate import FIRE


def _proc_vision() -> FakeVision:
    # scripted per-clip step deltas: "<frame_token>|step" -> a Verdict whose evidence is the step
    return FakeVision(script={
        "f1|step": Verdict(satisfied=False, confidence=0.9, evidence="peel the onion"),
        "f2|step": Verdict(satisfied=False, confidence=0.9, evidence="slice the onion"),
        "f3|step": Verdict(satisfied=False, confidence=0.9, evidence="peel the onion"),  # paraphrase/dup
        "f4|step": Verdict(satisfied=False, confidence=0.9, evidence="season the beef"),
        "quiet|step": Verdict(satisfied=False, confidence=0.9, evidence=""),  # nothing happening
    })


def _proc_goal() -> Goal:
    return Goal(id="proc", condition="what are the steps to make the dish",
               say="answer ready", state={"type": "process"})


class TestLogAndSummarize(unittest.TestCase):
    def test_observe_appends_to_log_and_never_fires(self):
        a = Percept.create(vision=_proc_vision(), discover_plugins=False)
        a.add_goal(_proc_goal())

        async def go():
            ev = []
            for i, tok in enumerate(("f1", "f2", "f3", "quiet"), start=1):
                ev += await a.perceive_judged(tok, t=float(i))
            return ev

        events = asyncio.run(go())
        ctx = a.executor.ctx("proc")
        self.assertEqual(ctx.process_log, ["peel the onion", "slice the onion", "peel the onion"])  # empty skipped
        self.assertEqual(ctx.last_step_at, 3.0)  # the "quiet" frame (t=4) logged nothing
        self.assertEqual([e for e in events if e.action == FIRE], [])  # a process goal never fires per-clip

    def test_summarize_on_demand_dedups_and_wipe_is_optional(self):
        a = Percept.create(vision=_proc_vision(), discover_plugins=False)
        a.add_goal(_proc_goal())

        async def go():
            for i, tok in enumerate(("f1", "f2", "f3"), start=1):
                await a.perceive_judged(tok, t=float(i))
            peek = await a.summarize_process("proc", wipe=False)   # "what have you seen?" — no wipe
            still = list(a.executor.ctx("proc").process_log)
            final = await a.summarize_process("proc", wipe=True)   # end -> wipe
            return peek, still, final

        peek, still, final = asyncio.run(go())
        self.assertEqual(peek, "peel the onion; slice the onion")   # order-preserving dedup
        self.assertEqual(len(still), 3)                              # wipe=False kept the log
        self.assertEqual(final, "peel the onion; slice the onion")
        self.assertEqual(a.executor.ctx("proc").process_log, [])     # wipe=True cleared it


class TestEpisodeBoundary(unittest.TestCase):
    def test_boundary_fires_once_on_sustained_absence_then_rearms(self):
        a = Percept.create(vision=_proc_vision(), discover_plugins=False)
        a.add_goal(_proc_goal())

        async def go():
            for i, tok in enumerate(("f1", "f2", "f3"), start=1):  # steps at t=1,2,3
                await a.perceive_judged(tok, t=float(i))
            too_soon = await a.check_episode_boundary(t=3.5, inactivity_s=2.0)   # 0.5s quiet < 2 -> no
            ended = await a.check_episode_boundary(t=6.0, inactivity_s=2.0)      # 3s quiet -> FIRE
            again = await a.check_episode_boundary(t=7.0, inactivity_s=2.0)      # already fired + wiped -> no
            await a.perceive_judged("f4", t=8.0)                                 # a NEW step -> new episode
            ep2 = await a.check_episode_boundary(t=11.0, inactivity_s=2.0)       # 3s quiet -> FIRE again
            return too_soon, ended, again, ep2

        too_soon, ended, again, ep2 = asyncio.run(go())
        self.assertEqual(too_soon, [])
        self.assertEqual(len(ended), 1)                                  # one fire per episode
        self.assertEqual(ended[0].action, FIRE)
        self.assertEqual(ended[0].verdict.evidence, "peel the onion; slice the onion")  # the consolidated answer
        self.assertEqual(a.executor.ctx("proc").process_log, [])         # wiped on fire
        self.assertEqual(again, [])                                       # refractory + empty log
        self.assertEqual(len(ep2), 1)                                    # re-armed for the next episode
        self.assertEqual(ep2[0].verdict.evidence, "season the beef")

    def test_boundary_ignores_a_non_process_goal(self):
        a = Percept.create(vision=_proc_vision(), discover_plugins=False)
        a.add_goal(Goal(id="watch", condition="someone at the door", say="hi", state={"type": "watch"}))

        out = asyncio.run(a.check_episode_boundary(t=100.0, inactivity_s=1.0))
        self.assertEqual(out, [])  # only type=process goals participate


class TestBoundaryThroughRuntimeWorker(unittest.TestCase):
    def test_worker_emits_the_consolidated_answer_as_an_incident(self):
        from percept.edge.tier0 import gray_frame

        still = gray_frame(8, 8, 0)
        moved = gray_frame(8, 8, 255)
        # (t, gate_frame, judge_frame): the gate escalates on CHANGE, so alternate still/moved to observe
        # both steps; then hold still (quiet) so the boundary check fires the consolidated answer.
        source = [
            (0.0, still, "quiet"),  # prime
            (1.0, moved, "f1"),     # change -> escalate -> log "peel the onion" (last_step_at=1)
            (2.0, still, "f2"),     # change -> escalate -> log "slice the onion" (last_step_at=2)
            (3.0, still, "quiet"),  # no change -> no escalate; boundary 3-2=1 < 1.5 -> no
            (4.0, still, "quiet"),  # boundary 4-2=2 > 1.5 -> FIRE the consolidated answer
        ]
        a = Percept.create(vision=_proc_vision(), discover_plugins=False)
        a.add_goal(_proc_goal())

        incidents = asyncio.run(
            self._collect(a, source, episode_inactivity_s=1.5)
        )
        self.assertEqual(len(incidents), 1)  # one episode-boundary incident, no per-clip fires
        self.assertEqual(incidents[0].verdict.evidence, "peel the onion; slice the onion")

    @staticmethod
    async def _collect(core, source, **kw):
        from percept.runtime import run
        return [inc async for inc in run(core, source=source, **kw)]


if __name__ == "__main__":
    unittest.main()
