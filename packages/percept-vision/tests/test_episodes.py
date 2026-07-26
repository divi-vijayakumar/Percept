"""L1: episode counting — new-vs-continuation dedup, window bound, Percept count goal (stub judge)."""

import unittest

from percept.adapters import Goal
from percept.agent import Percept
from percept.episodes import EpisodeLog

# Stub judges (NOT intelligence — they exercise the dedup logic deterministically).
ALWAYS_NEW = lambda summaries, exemplar: None  # noqa: E731
ALWAYS_CONTINUE = lambda summaries, exemplar: summaries[-1]["id"] if summaries else None  # noqa: E731


class TestEpisodeLog(unittest.TestCase):
    def test_sips_are_one_episode(self):
        log = EpisodeLog("g1")
        for t in (1.0, 2.0, 3.0):  # sip, sip, sip
            log.observe(t, "cup", ALWAYS_CONTINUE)
        self.assertEqual(log.count(), 1)

    def test_fresh_cup_increments(self):
        log = EpisodeLog("g1")
        log.observe(1.0, "cup", ALWAYS_NEW)
        log.observe(2.0, "cup", ALWAYS_CONTINUE)  # same cup
        ep, is_new = log.observe(3.0, "cup", ALWAYS_NEW)  # fresh cup
        self.assertTrue(is_new)
        self.assertEqual(log.count(), 2)

    def test_continuation_extends_last_seen(self):
        log = EpisodeLog("g1")
        ep, _ = log.observe(1.0, "cup", ALWAYS_NEW)
        same, is_new = log.observe(5.0, "cup", ALWAYS_CONTINUE)
        self.assertFalse(is_new)
        self.assertEqual(same.id, ep.id)
        self.assertEqual(same.last_seen, 5.0)

    def test_window_bound_prunes_old_episodes(self):
        log = EpisodeLog("g1", window_s=10.0)
        log.observe(1.0, "cup", ALWAYS_NEW)
        # a long gap past the window -> the old episode falls out of "today's cups"
        self.assertEqual(log.count(now=100.0), 0)

    def test_max_episodes_bound(self):
        log = EpisodeLog("g1", max_episodes=2)
        for t in range(5):
            log.observe(float(t), f"cup{t}", ALWAYS_NEW)
        self.assertEqual(log.count(), 2)

    def test_max_episodes_drop_is_traced_with_reason(self):
        # ADR-001 §9 / the implementation: the episode buffer's count bound must never drop silently.
        dropped = []
        log = EpisodeLog("g1", max_episodes=2, on_drop=lambda ep, reason: dropped.append((ep.id, reason)))
        for t in range(4):
            log.observe(float(t), f"cup{t}", ALWAYS_NEW)
        self.assertEqual(log.count(), 2)
        self.assertEqual(dropped, [("g1:ep1", "max_episodes"), ("g1:ep2", "max_episodes")])

    def test_window_bound_drop_is_traced_with_reason(self):
        dropped = []
        log = EpisodeLog("g1", window_s=10.0, on_drop=lambda ep, reason: dropped.append((ep.id, reason)))
        log.observe(1.0, "cup", ALWAYS_NEW)
        self.assertEqual(log.count(now=100.0), 0)
        self.assertEqual(dropped, [("g1:ep1", "window_expired")])

    def test_default_drop_is_traced_via_log_not_silent(self):
        # No on_drop supplied: the default path must still observably log the drop.
        log = EpisodeLog("g1", max_episodes=1)
        with self.assertLogs("percept.episodes", level="INFO") as cm:
            log.observe(1.0, "cupA", ALWAYS_NEW)
            log.observe(2.0, "cupB", ALWAYS_NEW)
        self.assertTrue(any("episode.dropped" in line for line in cm.output))


class TestPerceptCountEvent(unittest.TestCase):
    def _agent(self):
        a = Percept.create(discover_plugins=False)
        a.add_goal(
            Goal(
                id="cups",
                condition="the user finished a cup of coffee",
                say="that's another cup",
                state={"type": "count", "count": 0},
            )
        )
        return a

    def test_count_event_updates_goal_state(self):
        a = self._agent()
        _, n1 = a.count_event("cups", "cupA", ALWAYS_NEW, t=1.0)
        a.count_event("cups", "cupA", ALWAYS_CONTINUE, t=2.0)  # same cup, no increment
        _, n3 = a.count_event("cups", "cupB", ALWAYS_NEW, t=3.0)  # new cup
        self.assertTrue(n1 and n3)
        self.assertEqual(a.goals.get("cups").state["count"], 2)
        self.assertEqual(a.metrics.get("episode.new"), 2)
        self.assertEqual(a.metrics.get("episode.continuation"), 1)


if __name__ == "__main__":
    unittest.main()
