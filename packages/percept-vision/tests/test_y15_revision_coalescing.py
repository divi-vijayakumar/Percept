"""— revision coalescing; a 24/7 confirm heartbeat must not be an unbounded log.

The stress-test defect (found by the seven-domain eval, not either domain probe): a long-running
watch confirms the SAME finding on every confident tick, and the naive episode timeline appended a
``Revision`` per confirmation. Over a 24/7 watch that is UNBOUNDED — N identical confirmations, N
rows, forever. A re-statement of an unchanged fact is not a state change and must not grow the log.

The fix, asserted here (the ``eval/synthetic/idempotency/`` spec): ``confirmed`` COALESCES IN PLACE —
it advances ``last_confirmed`` / ``last_seen`` and appends NOTHING, so a run of N unchanged
confirmations produces exactly ONE row (the ``opened``). Only STATE-CHANGING kinds append, and the
per-episode timeline is bounded with a TRACED drop on overflow (ADR-001 §9: never silent).
"""

import logging
import unittest

from percept.episodes import (
    REVISION_KINDS,
    STATE_CHANGING_REVISION_KINDS,
    CLOSED,
    EpisodeLog,
    EpisodePolicy,
    Revision,
)


def _log(gap=1000.0, **kw):
    return EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=gap), **kw)


class TestConfirmHeartbeatCoalesces(unittest.TestCase):
    """The card: N unchanged confirmations must produce exactly ONE row — the open."""

    def test_run_of_confirmations_does_not_grow_the_timeline(self):
        log = _log()
        d = log.observe_decision(0.0, "spill:aisle-4", subject_key="spill:aisle-4")
        ep = d.episode
        self.assertEqual([r.kind for r in ep.revisions], ["opened"])
        # 500 confident ticks of the SAME finding — a 24/7 heartbeat.
        for t in range(1, 501):
            again = log.observe_decision(float(t), "spill:aisle-4", subject_key="spill:aisle-4")
            self.assertFalse(again.is_new)
            self.assertIs(again.episode, ep)
        # The timeline never grew past the single open; the heartbeat coalesced.
        self.assertEqual(len(ep.revisions), 1)
        self.assertEqual(ep.revisions[0].kind, "opened")
        self.assertEqual(log.count(), 1)
        # ...but last_seen / last_confirmed DID advance in place — the heartbeat is not lost.
        self.assertEqual(ep.last_seen, 500.0)
        self.assertEqual(ep.last_confirmed, 500.0)
        self.assertEqual(ep.observations, 501)

    def test_record_revision_confirmed_appends_nothing(self):
        log = _log()
        ep = log.observe_decision(0.0, "s", subject_key="s").episode
        out = log.record_revision(ep, "confirmed", t=9.0)
        self.assertIsNone(out)                       # coalesced — no row minted
        self.assertEqual(len(ep.revisions), 1)       # still just the open
        self.assertEqual(ep.last_confirmed, 9.0)
        self.assertEqual(ep.last_seen, 9.0)

    def test_confirmed_never_moves_last_seen_backwards(self):
        log = _log()
        ep = log.observe_decision(10.0, "s", subject_key="s").episode
        log.record_revision(ep, "confirmed", t=3.0)  # a late/out-of-order tick
        self.assertEqual(ep.last_seen, 10.0)         # not dragged backwards
        self.assertEqual(ep.last_confirmed, 3.0)     # but recorded as the confirm ts


class TestStateChangingKindsAppend(unittest.TestCase):
    def test_each_state_change_appends_one_row(self):
        log = _log()
        ep = log.observe_decision(0.0, "s", subject_key="s").episode
        r1 = log.record_revision(ep, "updated", t=1.0, value="v2")
        r2 = log.record_revision(ep, "retracted", t=2.0)
        self.assertIsInstance(r1, Revision)
        self.assertIsInstance(r2, Revision)
        self.assertEqual([r.kind for r in ep.revisions], ["opened", "updated", "retracted"])
        self.assertEqual(ep.revisions[1].value, "v2")

    def test_reopen_records_a_state_change(self):
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1000.0, allow_reopen=True))
        d1 = log.observe_decision(0.0, "s", subject_key="s")
        log.close(d1.episode.id, t=1.0)
        # A same-subject observation while still inside the gap reopens the SAME episode.
        d2 = log.observe_decision(100.0, "s", subject_key="s")
        self.assertFalse(d2.is_new)
        self.assertEqual(d2.reason, "reopened")
        self.assertEqual([r.kind for r in d2.episode.revisions], ["opened", "closed", "opened"])

    def test_close_records_a_closed_row(self):
        log = _log()
        d = log.observe_decision(0.0, "s", subject_key="s")
        log.annotate(d.episode, incident_id="INC-1")
        log.close(d.episode.id, t=5.0)
        self.assertEqual(d.episode.status, CLOSED)
        self.assertEqual(d.episode.revisions[-1].kind, "closed")
        self.assertEqual(d.episode.revisions[-1].incident_id, "INC-1")

    def test_unknown_kind_raises_loudly(self):
        log = _log()
        ep = log.observe_decision(0.0, "s", subject_key="s").episode
        with self.assertRaises(ValueError):
            log.record_revision(ep, "made_up", t=1.0)  # type: ignore[arg-type]


class TestVocabulariesAreClosed(unittest.TestCase):
    def test_confirmed_is_not_a_state_changing_kind(self):
        self.assertIn("confirmed", REVISION_KINDS)
        self.assertNotIn("confirmed", STATE_CHANGING_REVISION_KINDS)

    def test_state_changing_is_the_rest_of_the_vocabulary(self):
        self.assertEqual(
            set(STATE_CHANGING_REVISION_KINDS), set(REVISION_KINDS) - {"confirmed"}
        )
        self.assertEqual(
            set(REVISION_KINDS),
            {"opened", "confirmed", "updated", "retracted", "closed", "expired", "unobserved_gap"},
        )


class TestBoundedWithTracedDrop(unittest.TestCase):
    """The timeline is capped; a bound-driven drop is never silent (ADR-001 §9)."""

    def test_cap_drops_oldest_and_calls_the_hook(self):
        dropped: list = []
        log = _log(max_revisions=3, on_revision_drop=lambda ep, rev: dropped.append((ep.id, rev.kind)))
        ep = log.observe_decision(0.0, "s", subject_key="s").episode  # 1 row: opened
        log.record_revision(ep, "updated", t=1.0)                     # 2
        log.record_revision(ep, "updated", t=2.0)                     # 3 (at cap)
        self.assertEqual(dropped, [])
        log.record_revision(ep, "retracted", t=3.0)                   # overflow -> drop oldest
        self.assertEqual(len(ep.revisions), 3)
        self.assertEqual(dropped, [(ep.id, "opened")])               # the oldest fell out, traced
        self.assertEqual([r.kind for r in ep.revisions], ["updated", "updated", "retracted"])

    def test_drop_is_logged_when_no_hook_is_wired(self):
        log = _log(max_revisions=1)
        ep = log.observe_decision(0.0, "s", subject_key="s").episode  # opened, at cap
        with self.assertLogs("percept.episodes", level=logging.INFO) as cm:
            log.record_revision(ep, "updated", t=1.0)                # overflow with no hook
        self.assertTrue(any("revision_dropped" in line for line in cm.output))

    def test_confirmations_never_approach_the_cap(self):
        # A heartbeat coalesces, so no number of confirmations can overflow the revision cap.
        log = _log(max_revisions=2)
        ep = log.observe_decision(0.0, "s", subject_key="s").episode
        for t in range(1, 1000):
            log.record_revision(ep, "confirmed", t=float(t))
        self.assertEqual(len(ep.revisions), 1)


if __name__ == "__main__":
    unittest.main()
