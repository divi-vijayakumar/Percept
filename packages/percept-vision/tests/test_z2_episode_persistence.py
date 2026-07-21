"""Card Z2 — episode persistence is SESSION-ONLY for v1.4, and that decision is ENFORCED + OBSERVABLE.

The founder decision (card Y9, merged) is that episode/counting memory does NOT survive a restart:
``Percept.snapshot()`` carries goals/gate/entities/action_log and deliberately NOT the ``episodes``
map. Documented is not enough — two paying client teams read percept 0.7.0 as silent/unobservable.
This card makes the decision hold in CODE:

(a) ``snapshot()`` must not carry ``episodes`` — a future accidental add TRIPS a test, forcing the
    policy to be revisited deliberately rather than flipped by accident.
(b) ``restore()`` with any active counting goal must emit ONE loud ``episode_memory_reset`` health
    event (the named Y9 failure mode: an operator under-counts after a redeploy with nothing saying
    why). Percept queues; ``percept.runtime.run`` drains it into ``on_health``, once, at startup.
(c) ``Episode.to_dict`` / ``Episode.from_dict`` exist as the forward seam a later phase turns ON —
    but are NOT wired for v1.4 (snapshot/restore never call them).

If a later phase genuinely persists episodes, the failing tests here are the signal to go rewrite the
policy — NOT to loosen an assertion.
"""

from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame
from percept.episodes import Episode, EpisodeLog
from percept.persistence import InMemoryPersistence
from percept.runtime import run

STILL = gray_frame(8, 8, 0)
SRC = [(float(i), STILL, "x") for i in range(3)]


def _core() -> Percept:
    return Percept.create(
        vision=FakeVision(script={"x": Verdict(satisfied=False, confidence=0.9)}),
        discover_plugins=False,
    )


def _count_goal(gid: str = "cups") -> Goal:
    return Goal(id=gid, condition="a cup on the desk", say="counted",
                state={"type": "count", "count": 0})


class SnapshotDoesNotCarryEpisodes(unittest.TestCase):
    """(a) the session-only decision, enforced against the serialized shape."""

    def test_snapshot_omits_episodes_key(self):
        p = _core()
        p.add_goal(_count_goal())
        p.episodes["cups"] = EpisodeLog("cups")  # populate the very state that must not ride along
        snap = p.snapshot()
        self.assertNotIn("episodes", snap)
        # Belt-and-braces: no EpisodeLog may hide inside a nested structure either.
        self.assertNotIn("EpisodeLog", repr(snap))

    def test_a_future_accidental_add_would_be_caught_here(self):
        """The whole point: this key set is the tripwire. If snapshot() grows ``episodes``, this fails
        and the policy MUST be revisited deliberately (persisted representation + doc rewrite)."""
        snap = _core().snapshot()
        self.assertEqual(snap.get("episodes", "__absent__"), "__absent__")


class RestoreAnnouncesTheReset(unittest.TestCase):
    """(b) the amnesia is SEEN, not silently discovered as a wrong count."""

    def test_restore_queues_episode_memory_reset_for_active_counting_goals(self):
        src = _core()
        src.add_goal(_count_goal())
        snap = src.snapshot()

        fresh = _core()  # a fresh process == a restart; empty episode map
        fresh.restore(snap)

        queued = fresh._restore_health_log
        self.assertEqual(len(queued), 1)
        kind, data = queued[0]
        self.assertEqual(kind, "episode_memory_reset")
        self.assertEqual(data["counting_goals"], 1)
        self.assertEqual(data["goal_ids"], ["cups"])
        self.assertIn("session", data["reason"])

    def test_no_reset_event_when_no_counting_goal_is_active(self):
        """A pure watch goal never used the episode map — nothing to announce, no false alarm."""
        src = _core()
        src.add_goal(Goal(id="zone", condition="a person is here", say="alert"))
        fresh = _core()
        fresh.restore(src.snapshot())
        self.assertEqual(fresh._restore_health_log, [])

    def test_runtime_drains_the_reset_into_on_health(self):
        store = InMemoryPersistence()
        c1 = _core()
        c1.add_goal(_count_goal())
        asyncio.run(_drain(c1, persistence=store))  # session 1: arm + save

        c2 = _core()  # session 2: restart
        events: list = []
        asyncio.run(_drain(c2, persistence=store, restore=True, on_health=events.append))

        kinds = [e.kind for e in events]
        self.assertEqual(kinds.count("episode_memory_reset"), 1, "emitted exactly once, at startup")
        ev = next(e for e in events if e.kind == "episode_memory_reset")
        self.assertEqual(ev.data["goal_ids"], ["cups"])

    def test_default_watch_run_emits_no_reset(self):
        """No persistence/restore -> no reset event; the health channel stays byte-for-byte quiet."""
        store = InMemoryPersistence()
        c1 = _core()
        c1.add_goal(Goal(id="zone", condition="a person is here", say="alert"))
        asyncio.run(_drain(c1, persistence=store))
        c2 = _core()
        events: list = []
        asyncio.run(_drain(c2, persistence=store, restore=True, on_health=events.append))
        self.assertNotIn("episode_memory_reset", [e.kind for e in events])


class TheForwardSeamExistsButIsNotWired(unittest.TestCase):
    """(c) Episode.to_dict/from_dict — present so a later phase flips persistence ON cleanly."""

    def test_episode_round_trips_through_dict(self):
        ep = Episode(id="e1", goal_id="cups", started_at=10.0, last_seen=42.0,
                     exemplar={"box": [0, 0, 1, 1]}, representative={"box": [1, 1, 2, 2]},
                     status="open", subject_key="cup-A", observations=3)
        back = Episode.from_dict(ep.to_dict())
        self.assertEqual(back, ep)

    def test_stream_time_fields_survive_the_seam(self):
        """started_at/last_seen are stream-time instants — they must be carried verbatim so a later
        phase can rebase them like deadlines (unlike a wall-vs-monotonic dwell base)."""
        d = Episode(id="e", goal_id="g", started_at=100.5, last_seen=200.25).to_dict()
        self.assertEqual((d["started_at"], d["last_seen"]), (100.5, 200.25))

    def test_seam_is_not_wired_into_snapshot_or_restore(self):
        """The seam must stay dormant in v1.4: snapshot/restore never touch to_dict/from_dict, so a
        populated episode map still does not survive a round-trip."""
        src = _core()
        src.add_goal(_count_goal())
        src.episodes["cups"] = EpisodeLog("cups")
        fresh = _core()
        fresh.restore(src.snapshot())
        self.assertEqual(fresh.episodes, {}, "episodes stay session-only — the seam is not wired")


async def _drain(core, **kw):
    return [inc async for inc in run(core, source=SRC, **kw)]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
