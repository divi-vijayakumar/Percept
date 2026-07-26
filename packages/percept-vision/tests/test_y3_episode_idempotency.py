"""— episode-owned idempotency; the gate bit must not be accidental per-episode memory.

The failure this suite fixes: the gate is ONE BIT PER KEY, but "have I already reported THIS episode?" is a
property of the EPISODE. The A->B->A trace only worked because B happened to leave the shared gate key
in a fired state — if B re-arms the key and then never fires (dwell/ASK), A's unchanged revisit rides
the wrongly-armed edge and fires a second time. And the same coupling makes a *housekeeping* event —
a gate-key TTL eviction while the episode is still open — leak into user-visible behaviour.

The acceptance spec is the two synthetic fixture trees (``eval/synthetic/continuation/`` and
``eval/synthetic/idempotency/``); this module encodes their assertions so they hold in a worktree that
does not carry the gitignored fixtures. ``percept/gate.py`` is asserted mechanically unchanged: the
episode reaches the gate only as ``rearm()`` / ``evaluate()`` calls, never as new gate state.
"""

import unittest

from percept.adapters import Goal, Verdict
from percept.behavior import (
    BehaviorExecutor,
    BehaviorProgram,
    EPISODE_MATCHES,
    EPISODE_MUTATIONS,
    EpisodeFoldResult,
    EpisodeIncidentLedger,
    episode_fold,
    revision_id,
)
from percept.episodes import EpisodeLog, EpisodePolicy
from percept.gate import Gate

_YES = Verdict(satisfied=True, confidence=1.0, evidence="scripted true")


def _goal(**state):
    return Goal(id="g", condition="c", say="fired", state={"type": "watch", **state})


# --------------------------------------------------------------------------------------------------
# The pure fold — the record the implementation recomputes in shadow. Side-effect free, closed vocabularies.
# --------------------------------------------------------------------------------------------------


class TestEpisodeFoldPure(unittest.TestCase):
    def _log(self, gap=10.0):
        return EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=gap))

    def test_unbound_falls_back_to_gate_only(self):
        f = episode_fold(None)
        self.assertEqual(f.match, "unbound")
        self.assertIsNone(f.episode_id)
        self.assertTrue(f.should_evaluate_gate)   # legacy gate-only behaviour
        self.assertFalse(f.should_rearm)
        self.assertEqual(f.mutation, "none")

    def test_ambiguous_falls_back_to_gate_only(self):
        log = self._log()
        d = log.observe_decision(0.0, "s", subject_key="s")
        f = episode_fold(d, ambiguous=True)
        self.assertEqual(f.match, "ambiguous")
        self.assertTrue(f.should_evaluate_gate)
        self.assertFalse(f.should_rearm)

    def test_new_episode_evaluates_and_rearms(self):
        log = self._log()
        d = log.observe_decision(0.0, "s", subject_key="s")
        f = episode_fold(d, emitted=0)
        self.assertEqual(f.match, "new")
        self.assertEqual(f.mutation, "none")
        self.assertTrue(f.should_evaluate_gate)
        self.assertTrue(f.should_rearm)      # its own edge, immune to a shared/disarmed key
        self.assertIsNone(f.supersedes)

    def test_continuation_unemitted_keeps_evaluating(self):
        log = self._log()
        log.observe_decision(0.0, "s", subject_key="s")
        d = log.observe_decision(1.0, "s", subject_key="s")
        self.assertFalse(d.is_new)
        f = episode_fold(d, emitted=0)   # opened but never fired (e.g. still in dwell)
        self.assertEqual(f.match, "continuation")
        self.assertTrue(f.should_evaluate_gate)
        self.assertFalse(f.should_rearm)

    def test_continuation_after_emission_is_idempotent(self):
        # THE card: an already-emitted episode, re-confirmed unchanged, does NOT re-enter the gate.
        log = self._log()
        log.observe_decision(0.0, "s", subject_key="s")
        d = log.observe_decision(1.0, "s", subject_key="s")
        f = episode_fold(d, emitted=1)
        self.assertEqual(f.mutation, "confirmed")
        self.assertFalse(f.should_evaluate_gate)  # the gate is never consulted
        self.assertFalse(f.should_rearm)

    def test_accepted_update_supersedes_and_may_rearm(self):
        log = self._log()
        log.observe_decision(0.0, "s", subject_key="s")
        d = log.observe_decision(1.0, "s", subject_key="s")
        f = episode_fold(d, emitted=1, mutation="updated")
        self.assertEqual(f.mutation, "updated")
        self.assertTrue(f.should_evaluate_gate)
        self.assertTrue(f.should_rearm)
        self.assertEqual(f.supersedes, revision_id(d.episode.id, 1))

    def test_update_before_first_emission_is_just_more_evidence(self):
        log = self._log()
        log.observe_decision(0.0, "s", subject_key="s")
        d = log.observe_decision(1.0, "s", subject_key="s")
        f = episode_fold(d, emitted=0, mutation="updated")
        self.assertTrue(f.should_evaluate_gate)
        self.assertFalse(f.should_rearm)      # nothing emitted to supersede yet
        self.assertIsNone(f.supersedes)

    def test_closed_stops_evaluation(self):
        log = self._log()
        log.observe_decision(0.0, "s", subject_key="s")
        d = log.observe_decision(1.0, "s", subject_key="s")
        f = episode_fold(d, emitted=1, mutation="closed")
        self.assertFalse(f.should_evaluate_gate)
        self.assertFalse(f.should_rearm)

    def test_retracted_rearms_for_a_genuine_return(self):
        log = self._log()
        log.observe_decision(0.0, "s", subject_key="s")
        d = log.observe_decision(1.0, "s", subject_key="s")
        f = episode_fold(d, emitted=1, mutation="retracted")
        self.assertTrue(f.should_rearm)
        self.assertEqual(f.supersedes, revision_id(d.episode.id, 1))

    def test_result_vocabularies_are_closed(self):
        with self.assertRaises(ValueError):
            EpisodeFoldResult("e", "made_up", "none", False, True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            EpisodeFoldResult("e", "new", "bogus", False, True)  # type: ignore[arg-type]
        self.assertEqual(
            set(EPISODE_MATCHES),
            {"new", "continuation", "unbound", "ambiguous", "unknown"},  # Y4 added "unknown"
        )
        self.assertEqual(
            set(EPISODE_MUTATIONS), {"confirmed", "updated", "retracted", "closed", "none"}
        )


class TestIncidentLedger(unittest.TestCase):
    def test_records_and_mints_revisions(self):
        led = EpisodeIncidentLedger()
        self.assertEqual(led.emitted("g:ep1"), 0)
        self.assertIsNone(led.revision("g:ep1"))
        self.assertEqual(led.record("g:ep1"), "g:ep1#r1")
        self.assertEqual(led.emitted("g:ep1"), 1)
        self.assertEqual(led.record("g:ep1"), "g:ep1#r2")
        self.assertEqual(led.revision("g:ep1"), "g:ep1#r2")

    def test_episodes_do_not_collide(self):
        led = EpisodeIncidentLedger()
        led.record("g:ep1")
        self.assertEqual(led.emitted("g:ep2"), 0)  # a different episode is independent


# --------------------------------------------------------------------------------------------------
# The wiring — fire_judged honours the fold; the gate stays one bit per key.
# --------------------------------------------------------------------------------------------------


class _SpyGate(Gate):
    """gate.py stays MECHANICALLY UNCHANGED — the episode reaches it only as rearm()/evaluate()."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rearms: list = []

    def rearm(self, key):
        self.rearms.append(key)
        super().rearm(key)


def _drive(gate, ex, goal, prog, log, stream, *, sweeps=None):
    """Replay ``stream`` = [(t, subject, mutation)] through the executor with an EpisodeLog + fold.

    ``sweeps`` maps a step index -> a ``t`` at which to touch an UNRELATED gate key first, forcing a
    TTL sweep that evicts the (idle) episode key just like real background traffic would. Returns the
    per-step actions and the emitted-incident count.
    """
    actions, incidents = [], 0
    key = (goal.id, "scene")  # ONE shared gate key — the coupling Y3 has to break
    for i, (t, subject, mutation) in enumerate(stream):
        if sweeps and i in sweeps:
            gate.evaluate((goal.id, "__housekeeping__"), Verdict(False, 1.0), now=sweeps[i])
        d = log.observe_decision(t, subject, subject_key=subject)
        fold = ex.fold_episode(d, mutation=mutation)
        fired = ex.fire_judged(goal, _YES, key, prog, t=t, fold=fold)
        if fired is None:
            actions.append("silent")
        else:
            actions.append(fired.action)
            incidents += 1
    return actions, incidents


class TestIdempotencyFixture(unittest.TestCase):
    """eval/synthetic/idempotency/ — re-confirming an already-reported episode is not a new incident."""

    def test_unchanged_confirmation_does_not_reincident(self):
        gate, ex = _SpyGate(), None
        ex = BehaviorExecutor(gate)
        goal, prog = _goal(), BehaviorProgram(sense="judge")
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1000.0))
        actions, incidents = _drive(
            gate, ex, goal, prog, log,
            [(0.0, "spill:aisle-4", "confirmed"),
             (5.0, "spill:aisle-4", "confirmed"),
             (10.0, "spill:aisle-4", "confirmed")],
        )
        self.assertEqual(actions, ["fire", "silent", "silent"])
        self.assertEqual(incidents, 1)
        self.assertEqual(log.count(), 1)  # one episode, open throughout

    def test_key_ttl_expiry_while_episode_open_does_not_refire(self):
        # THE invariant: a gate-key TTL sweep is housekeeping. With the key evicted mid-episode the
        # gate alone would read the still-open episode as a fresh rising edge and fire again; the
        # episode-owned fold makes that impossible because it never re-enters the gate at all.
        gate = _SpyGate(key_ttl_s=60.0)
        ex = BehaviorExecutor(gate)
        goal, prog = _goal(), BehaviorProgram(sense="judge")
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1000.0))
        actions, incidents = _drive(
            gate, ex, goal, prog, log,
            [(0.0, "spill:aisle-4", "confirmed"),
             (61.0, "spill:aisle-4", "confirmed"),
             (120.0, "spill:aisle-4", "confirmed")],
            sweeps={1: 61.0},  # background traffic sweeps the idle spill key before step 1
        )
        self.assertEqual(actions, ["fire", "silent", "silent"])
        self.assertEqual(incidents, 1)

    def test_without_the_fold_the_evicted_key_DOES_refire(self):
        # The counter-proof that the fold is load-bearing: same stream, gate only (no fold). The TTL
        # sweep drops the armed key, so the second confirmation reads as a new rising edge and fires.
        gate = Gate(key_ttl_s=60.0)
        ex = BehaviorExecutor(gate)
        goal, prog = _goal(), BehaviorProgram(sense="judge")
        key = (goal.id, "scene")
        fires = 0
        for i, t in enumerate((0.0, 61.0, 120.0)):
            if i == 1:
                gate.evaluate((goal.id, "__housekeeping__"), Verdict(False, 1.0), now=61.0)
            if ex.fire_judged(goal, _YES, key, prog, t=t) is not None:
                fires += 1
        self.assertEqual(fires, 2)  # the leak Y3 exists to close


class TestAbaContinuationFixture(unittest.TestCase):
    """eval/synthetic/continuation/ at the fire path — a revisit must not mint a spurious third fire,
    and B failing its bar must not leave an edge for A to ride."""

    #: A fires (YES). B is a new episode whose read is low-confidence -> ASK, so it never fires but
    #: still re-arms the SHARED scene key. A revisits with a clean YES. This is the exact fragile
    #: trace from the card: "B ... becomes ASK and never fires, A's return is a continuation ...".
    _STREAM = [(0.0, "A", _YES), (1.0, "B", Verdict(True, 0.3)), (2.0, "A", _YES)]

    def _setup(self, prog):
        gate = _SpyGate()
        ex = BehaviorExecutor(gate)
        goal = _goal()
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=10.0))
        return gate, ex, goal, prog, log

    def test_b_becomes_ask_must_not_let_a_refire(self):
        prog = BehaviorProgram(sense="judge")
        gate, ex, goal, prog, log = self._setup(prog)
        key = (goal.id, "scene")
        actions = []
        for t, subject, verdict in self._STREAM:
            d = log.observe_decision(t, subject, subject_key=subject)
            fold = ex.fold_episode(d, mutation="confirmed")
            fired = ex.fire_judged(goal, verdict, key, prog, t=t, fold=fold)
            actions.append("silent" if fired is None else fired.action)
        # A fires; B is ASK (re-arms the shared key but emits nothing); A's revisit CONTINUES an
        # already-emitted episode and is idempotently suppressed even though the key is armed.
        self.assertEqual(actions, ["fire", "ask", "silent"])
        self.assertIn(("g", "scene"), gate.rearms)  # B really did re-arm the shared key

    def test_without_the_fold_bs_rearm_makes_a_refire(self):
        # The counter-proof: same trace, no fold. B's rearm leaves the shared key armed, so A's third
        # observation rides it and fires a SECOND time — the coupling Y3 removes.
        gate = _SpyGate()
        ex = BehaviorExecutor(gate)
        goal, prog = _goal(), BehaviorProgram(sense="judge")
        key = (goal.id, "scene")
        actions = []
        for t, subject, verdict in self._STREAM:
            # Emulate the legacy executor: the episode boundary reached the gate only as a rearm on a
            # NEW episode, with no episode-owned idempotency guard.
            if subject == "B":
                gate.rearm(key)
            fired = ex.fire_judged(goal, verdict, key, prog, t=t)
            actions.append("silent" if fired is None else fired.action)
        self.assertEqual(actions, ["fire", "ask", "fire"])  # A double-fires — the bug

    def test_a_genuine_second_occurrence_after_the_gap_still_fires(self):
        # Separation must not SUPPRESS real repeats: the same subject after the continuity gap is a new
        # episode and DOES fire again.
        prog = BehaviorProgram(sense="judge")
        gate, ex, goal, prog, log = self._setup(prog)
        actions, incidents = _drive(
            gate, ex, goal, prog, log,
            [(0.0, "A", "confirmed"), (100.0, "A", "confirmed")],
        )
        self.assertEqual(actions, ["fire", "fire"])
        self.assertEqual(incidents, 2)


class TestByteForByteWhenNoFold(unittest.TestCase):
    def test_fold_none_is_the_legacy_path(self):
        # The additive guarantee: with no fold, fire_judged is exactly the legacy gate-only behaviour.
        gate = Gate()
        ex = BehaviorExecutor(gate)
        goal, prog = _goal(), BehaviorProgram(sense="judge")
        key = (goal.id, "e")
        first = ex.fire_judged(goal, _YES, key, prog, t=0.0)
        second = ex.fire_judged(goal, _YES, key, prog, t=1.0)
        self.assertIsNotNone(first)          # rising edge fires
        self.assertIsNone(second)            # already fired, gate bit alone suppresses
        self.assertEqual(gate._armed.get(key), False)

    def test_gate_never_learns_about_episodes(self):
        gate = Gate()
        for forbidden in ("episode", "episodes", "incidents", "_incidents", "fold", "supersedes"):
            self.assertFalse(hasattr(gate, forbidden), f"gate grew {forbidden!r}")


class TestSuppressionIsObservable(unittest.TestCase):
    def test_idempotent_suppression_is_metered(self):
        seen = []

        class _M:
            def incr(self, k):
                seen.append(k)

        gate = Gate()
        ex = BehaviorExecutor(gate, metrics=_M())
        goal, prog = _goal(), BehaviorProgram(sense="judge")
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1000.0))
        key = (goal.id, "scene")
        for t, subject in ((0.0, "s"), (1.0, "s")):
            d = log.observe_decision(t, subject, subject_key=subject)
            ex.fire_judged(goal, _YES, key, prog, t=t, fold=ex.fold_episode(d))
        self.assertIn("gate.idempotent_suppressed", seen)


if __name__ == "__main__":
    unittest.main()
