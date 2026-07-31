"""— state consistency + intervals that do NOT silently forward-fill.

Two things land here, both from the client-viability review:

**§1 STATE CONSISTENCY.** ``EpisodeLog.transition`` is the SINGLE authoritative writer of
``status``/``t_end``; no other code path may assign them, and the module is scanned to prove it.
``t_end`` is derivable (not None iff terminal), a reopen clears it, and ``value``/``incident_id`` use
``None`` — never ``''`` — as their one absence sentinel.

**§2 INTERVALS.** ``started_at -> last_seen`` does NOT prove continuous coverage. Two observations 19s
apart are two points and one UNOBSERVED gap, not a 19-second interval; a densely observed run IS a
merged span; a judge-degraded stretch is a ``judge_degraded`` gap, never continued truth. The
``eval/synthetic/intervals`` corpus is the SPEC — the cases are driven straight off its generator so
the fixture and the implementation cannot disagree silently.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import get_args

from percept.episodes import (
    CLOSED,
    OPEN,
    Coverage,
    CoverageGap,
    CoverageKind,
    CoverageSpan,
    EpisodeLog,
    EpisodePolicy,
    Observation,
    interval_coverage,
)
from percept.harness.synthetic.corpora import intervals


def _uniform_cadence(steps: list[dict]) -> float | None:
    """The concern-authored validity horizon these fixtures model: a densely + uniformly sampled
    stream declares its sample cadence as the horizon; a sparse stream (the two-point case) authored
    NO horizon, so it must refuse to fill. This is a property of the fixture, computed here rather
    than hard-coded per case, so a regenerated corpus stays honest."""
    ts = sorted(s["t"] for s in steps)
    deltas = [round(b - a, 6) for a, b in zip(ts, ts[1:])]
    if len(deltas) >= 2 and len(set(deltas)) == 1:
        return deltas[0]
    return None


def _observations(steps: list[dict]) -> list[Observation]:
    out: list[Observation] = []
    for s in steps:
        v = s.get("verdict") or {}
        out.append(
            Observation(
                t=float(s["t"]),
                observed=bool(v.get("observed")),
                satisfied=bool(v.get("satisfied")),
                degraded=bool(v.get("degraded")),
            )
        )
    return out


class IntervalCorpusTests(unittest.TestCase):
    """Drive the interval builder off the intervals corpus generator (the SPEC)."""

    @classmethod
    def setUpClass(cls) -> None:
        stream, labels = intervals.build()
        cls.cases = stream["cases"]
        cls.expect = {c["id"]: c["expect"] for c in labels["cases"]}

    def _coverage_for(self, case: dict) -> Coverage:
        return interval_coverage(
            _observations(case["steps"]),
            observation_valid_for_s=_uniform_cadence(case["steps"]),
        )

    def test_every_case_matches_the_spec(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                cov = self._coverage_for(case)
                exp = self.expect[case["id"]]
                self.assertEqual(
                    [(s.start, s.end, s.kind) for s in cov.spans],
                    [(i["start"], i["end"], i["kind"]) for i in exp["intervals"]],
                )
                self.assertEqual(
                    [(g.start, g.end, g.reason) for g in cov.gaps],
                    [(g["start"], g["end"], g["reason"]) for g in exp["gaps"]],
                )
                self.assertEqual(cov.observed_total_s, exp["observed_total_s"])

    def test_two_sparse_observations_are_not_a_span(self):
        """The decisive catch: t=1 and t=20 is NOT a 19-second interval."""
        case = next(c for c in self.cases if c["id"] == "two_sparse_observations_are_not_a_span")
        cov = self._coverage_for(case)
        self.assertEqual(cov.observed_total_s, 0.0)
        self.assertNotEqual(cov.observed_total_s, 19.0)  # the forbidden forward-filled answer
        self.assertEqual([(g.start, g.end) for g in cov.gaps], [(1.0, 20.0)])
        self.assertEqual(cov.gaps[0].reason, "unobserved")

    def test_degraded_stretch_is_a_gap_not_continued_truth(self):
        case = next(c for c in self.cases if c["id"] == "degraded_judge_gap_is_not_continued_truth")
        cov = self._coverage_for(case)
        # NOT 10.0 (carrying the last true verdict across the outage) and there IS a gap (not a
        # confident false).
        self.assertEqual(cov.observed_total_s, 4.0)
        self.assertEqual([g.reason for g in cov.gaps], ["judge_degraded"])


class IntervalBuilderUnitTests(unittest.TestCase):
    def test_default_horizon_never_forward_fills(self):
        """No horizon supplied => every positive is its own point, everything between is a gap."""
        cov = interval_coverage(
            [Observation(t=0.0, observed=True, satisfied=True),
             Observation(t=100.0, observed=True, satisfied=True)]
        )
        self.assertEqual(cov.observed_total_s, 0.0)
        self.assertEqual([(s.start, s.end) for s in cov.spans], [(0.0, 0.0), (100.0, 100.0)])
        self.assertEqual([g.reason for g in cov.gaps], ["unobserved"])

    def test_trailing_observation_is_never_extended_past_itself(self):
        """A horizon merges backward-bounded gaps only; extending the LAST observation into the open
        future would itself be forward-filling."""
        cov = interval_coverage(
            [Observation(t=0.0, observed=True, satisfied=True),
             Observation(t=1.0, observed=True, satisfied=True)],
            observation_valid_for_s=5.0,
        )
        self.assertEqual([(s.start, s.end) for s in cov.spans], [(0.0, 1.0)])
        self.assertEqual(cov.observed_total_s, 1.0)  # NOT 1.0 + 5.0

    def test_no_positive_observations_is_empty_coverage(self):
        cov = interval_coverage([Observation(t=3.0, observed=False, degraded=True)])
        self.assertEqual(cov, Coverage(spans=(), gaps=(), observed_total_s=0.0))

    def test_unordered_input_is_tolerated(self):
        cov = interval_coverage(
            [Observation(t=2.0, observed=True, satisfied=True),
             Observation(t=0.0, observed=True, satisfied=True)],
            observation_valid_for_s=5.0,
        )
        self.assertEqual([(s.start, s.end) for s in cov.spans], [(0.0, 2.0)])

    def test_unobserved_vs_degraded_gap_reason(self):
        # A degraded tick inside the gap makes it judge_degraded; nothing inside makes it unobserved.
        degraded = interval_coverage(
            [Observation(t=0.0, observed=True, satisfied=True),
             Observation(t=1.0, observed=False, degraded=True),
             Observation(t=2.0, observed=True, satisfied=True)]
        )
        self.assertEqual([g.reason for g in degraded.gaps], ["judge_degraded"])
        silent = interval_coverage(
            [Observation(t=0.0, observed=True, satisfied=True),
             Observation(t=2.0, observed=True, satisfied=True)]
        )
        self.assertEqual([g.reason for g in silent.gaps], ["unobserved"])


class TestTheValidityHorizonBoundary(unittest.TestCase):
    """§2.5 — an observation landing EXACTLY ``observation_valid_for_s`` after the previous positive.

    The merge test is ``within(dt, observation_valid_for_s)``, so the boundary is INCLUSIVE:
    exactly-on-the-horizon continues the span. That is the documented side (the module header says
    an observation is valid *through* ``t + observation_valid_for_s``), and it is pinned here from
    both directions so a later ``<`` cannot silently halve every span at the boundary — and, since
    BUG-B-04-01, inclusive IN FACT rather than only when the arithmetic happens to be exact.
    """

    T_A, T_B = 0.0, 5.0  # dt is exactly 5.0, and 5.0 is exactly representable in binary float

    def _cov(self, horizon: "float | None", *, explicit: bool = True) -> Coverage:
        obs = [
            Observation(t=self.T_A, observed=True, satisfied=True),
            Observation(t=self.T_B, observed=True, satisfied=True),
        ]
        if not explicit:
            return interval_coverage(obs)
        return interval_coverage(obs, observation_valid_for_s=horizon)

    def test_exactly_on_the_horizon_continues_the_span(self):
        """dt == horizon merges: one span, no gap, and the whole 5s counts as accounted for."""
        cov = self._cov(5.0)
        self.assertEqual(cov.spans, (CoverageSpan(start=0.0, end=5.0, kind="observed"),))
        self.assertEqual(cov.gaps, ())
        self.assertEqual(cov.observed_total_s, 5.0)

    def test_one_millisecond_past_the_horizon_breaks_the_span(self):
        """The exclusive side. Both positives survive as ZERO-LENGTH spans — current behaviour, and
        the reason ``observed_total_s`` collapses to 0.0 rather than to the two instants' 5s apart."""
        cov = self._cov(4.999)
        self.assertEqual(
            cov.spans,
            (
                CoverageSpan(start=0.0, end=0.0, kind="observed"),
                CoverageSpan(start=5.0, end=5.0, kind="observed"),
            ),
        )
        self.assertEqual(cov.gaps, (CoverageGap(start=0.0, end=5.0, reason="unobserved"),))
        self.assertEqual(cov.observed_total_s, 0.0)

    def test_no_horizon_is_the_exclusive_side_at_every_separation(self):
        """``None`` (the shipped default) never merges, so it lands on exactly the exclusive ANSWER
        — the boundary knob is the ONLY thing that can move spans, gaps or the total.

        ``horizon_granted_s`` is deliberately excluded from the comparison: it is provenance, not
        an answer, and "the same coverage was reached under a different horizon" is precisely what
        it exists to tell a client."""
        def answer(cov):
            return (cov.spans, cov.gaps, cov.observed_total_s)

        exclusive = self._cov(4.999)
        self.assertEqual(answer(self._cov(None)), answer(exclusive))
        self.assertEqual(answer(self._cov(None, explicit=False)), answer(exclusive))
        self.assertEqual(self._cov(None), self._cov(None, explicit=False))  # default == explicit None
        self.assertNotEqual(self._cov(None), exclusive)  # ...and the provenance still differs

    def test_the_flip_happens_at_the_horizon_and_nowhere_else(self):
        """Sweep across dt=5.0: merged IFF horizon >= dt. Pins the inclusive side as a boundary, not
        as a lucky value — a ``<`` in the impl fails only the horizon==5.0 row."""
        for horizon, merged in [
            (0.0, False),
            (4.9, False),
            (4.999999, False),
            (5.0, True),  # the boundary itself
            (5.000001, True),
            (500.0, True),
        ]:
            with self.subTest(horizon=horizon):
                cov = self._cov(horizon)
                self.assertEqual(len(cov.spans), 1 if merged else 2)
                self.assertEqual(cov.gaps, () if merged else
                                 (CoverageGap(start=0.0, end=5.0, reason="unobserved"),))
                self.assertEqual(cov.observed_total_s, 5.0 if merged else 0.0)

    def test_a_degraded_tick_inside_the_merged_stretch_does_not_break_it(self):
        """The boundary is decided by the POSITIVES only: a degraded tick between them cannot force a
        gap, it can only name one that the horizon already refused to cover."""
        obs = [
            Observation(t=0.0, observed=True, satisfied=True),
            Observation(t=2.5, observed=False, degraded=True),
            Observation(t=5.0, observed=True, satisfied=True),
        ]
        merged = interval_coverage(obs, observation_valid_for_s=5.0)
        self.assertEqual(merged.gaps, ())
        self.assertEqual(merged.observed_total_s, 5.0)  # silence inside a valid horizon stays covered
        broken = interval_coverage(obs, observation_valid_for_s=4.999)
        self.assertEqual([g.reason for g in broken.gaps], ["judge_degraded"])

    def test_there_is_exactly_one_coverage_kind(self):
        """BUG-B-04-02, decided as retirement. ``CoverageKind`` declared a second literal,
        ``validity_horizon``, that NO code path in ``percept/`` ever produced — both ``CoverageSpan``
        construction sites hard-code ``observed`` — so every consumer branching on it was branching
        on a value it could not receive, and the synthetic schema validated a value that would then
        fail every comparison against real coverage.

        Emitting it for real was rejected: splitting a merged span would change span CARDINALITY and
        the meaning of ``observed_total_s`` for every merged run, a breaking read for any client
        summing spans. The literal is retired here and in ``harness/synthetic/schema.py``; the client
        question it was reaching for is answered additively by ``horizon_granted_s`` below.

        REWRITTEN from ``test_horizon_inferred_time_is_still_labelled_observed``, which asserted the
        unreachable literal was present in the type.
        """
        self.assertEqual(get_args(CoverageKind), ("observed",))
        cov = self._cov(5.0)
        self.assertEqual([s.kind for s in cov.spans], ["observed"])

    def test_coverage_reports_the_horizon_its_time_was_granted_under(self):
        """The additive answer. An observation is an INSTANT, so every second strictly inside a span
        was granted by the validity horizon rather than directly sampled — and a client deciding
        whether to quote ``observed_total_s`` as a duration needs to see how generous that grant
        was. 5s of coverage under a 5s horizon is one sample's benefit of the doubt."""
        self.assertEqual(self._cov(5.0).horizon_granted_s, 5.0)
        # No horizon => no grant, and necessarily no coverage: spans cannot exceed an instant.
        for cov in (self._cov(None), self._cov(None, explicit=False)):
            self.assertEqual(cov.horizon_granted_s, 0.0)
            self.assertEqual(cov.observed_total_s, 0.0)
        # It is provenance, not a measurement — reported even when nothing positive was observed.
        empty = interval_coverage(
            [Observation(t=3.0, observed=False, degraded=True)], observation_valid_for_s=2.0
        )
        self.assertEqual(empty.horizon_granted_s, 2.0)
        self.assertEqual(empty.observed_total_s, 0.0)

    def test_exact_horizon_holds_at_a_ten_hertz_cadence(self):
        """The same boundary, at a cadence whose step is NOT exactly representable (BUG-B-04-01).

        Every tick is a positive and the horizon is exactly the cadence, so the documented inclusive
        side says: one span, no gaps, 1.0s accounted for. A bare ``dt <= horizon`` fails this —
        ``0.4 - 0.3 == 0.10000000000000003 > 0.1`` while ``0.3 - 0.2 == 0.09999999999999998 < 0.1``,
        so coverage fragments on SOME steps and not others and the total lands on
        ``0.7999999999999998``. ``percept._time_compare.within`` closes the boundary."""
        ts = [round(i * 0.1, 10) for i in range(11)]  # 0.0 .. 1.0 at 10 Hz
        cov = interval_coverage(
            [Observation(t=t, observed=True, satisfied=True) for t in ts],
            observation_valid_for_s=0.1,
        )
        self.assertEqual(cov.gaps, ())
        self.assertEqual(cov.spans, (CoverageSpan(start=0.0, end=1.0, kind="observed"),))
        self.assertEqual(cov.observed_total_s, 1.0)

    def test_the_tolerance_is_a_boundary_nudge_not_a_slack_horizon(self):
        """The guard on the fix: ``within`` closes the boundary at nanosecond scale and nothing
        wider. A horizon a MILLISECOND short of the step still refuses to merge — the tolerance
        cannot be used to smuggle in coverage the client did not author."""
        cov = interval_coverage(
            [Observation(t=0.0, observed=True, satisfied=True),
             Observation(t=0.1, observed=True, satisfied=True)],
            observation_valid_for_s=0.099,
        )
        self.assertEqual(cov.observed_total_s, 0.0)
        self.assertEqual([g.reason for g in cov.gaps], ["unobserved"])


class StateConsistencyTests(unittest.TestCase):
    """§1: status/t_end have a single writer and cannot drift from each other."""

    def _log(self, **kw) -> EpisodeLog:
        return EpisodeLog("g", **kw)

    def test_open_episode_has_no_end(self):
        log = self._log()
        ep, is_new = log.observe(0.0, "x", subject_key="s")
        self.assertTrue(is_new)
        self.assertEqual(ep.status, OPEN)
        self.assertIsNone(ep.t_end)  # live episodes have no end

    def test_close_stamps_t_end_from_last_seen(self):
        log = self._log()
        ep, _ = log.observe(0.0, "x", subject_key="s")
        log.observe(5.0, "x", subject_key="s")  # last_seen advances to 5
        closed = log.close(ep.id)
        self.assertEqual(closed.status, CLOSED)
        self.assertEqual(closed.t_end, 5.0)  # the last moment it was observed, not invented later

    def test_close_with_explicit_t(self):
        log = self._log()
        ep, _ = log.observe(0.0, "x", subject_key="s")
        log.close(ep.id, t=9.0)
        self.assertEqual(log.get(ep.id).t_end, 9.0)

    def test_t_end_is_derivable_from_status(self):
        """t_end is not None IFF the status is terminal — they are never stored inconsistently."""
        log = self._log()
        ep, _ = log.observe(0.0, "x", subject_key="s")
        self.assertEqual(ep.t_end is not None, log.policy.is_terminal(ep.status))
        log.close(ep.id)
        self.assertEqual(ep.t_end is not None, log.policy.is_terminal(ep.status))

    def test_reopen_clears_t_end(self):
        policy = EpisodePolicy(continuity_gap_s=100.0, allow_reopen=True)
        log = self._log(policy=policy)
        ep, _ = log.observe(0.0, "x", subject_key="s")
        log.close(ep.id, t=1.0)
        self.assertIsNotNone(ep.t_end)
        again, is_new = log.observe(2.0, "x", subject_key="s")  # reopen
        self.assertFalse(is_new)
        self.assertIs(again, ep)
        self.assertEqual(ep.status, OPEN)
        self.assertIsNone(ep.t_end)  # a reopened episode has NO end again

    def test_transition_to_custom_terminal_status_stamps_end(self):
        policy = EpisodePolicy(terminal_statuses=frozenset({"resolved"}))
        log = self._log(policy=policy)
        ep, _ = log.observe(0.0, "x", subject_key="s")
        log.transition(ep, "resolved", t=4.0)
        self.assertEqual(ep.t_end, 4.0)
        # OPEN is non-terminal under this policy => t_end clears.
        log.transition(ep, OPEN, t=8.0)
        self.assertIsNone(ep.t_end)

    def test_value_and_incident_absence_is_none_never_empty_string(self):
        log = self._log()
        ep, _ = log.observe(0.0, "x", subject_key="s")
        self.assertIsNone(ep.value)
        self.assertIsNone(ep.incident_id)
        log.annotate(ep, value="", incident_id="")  # '' is absence, not presence
        self.assertIsNone(ep.value)
        self.assertIsNone(ep.incident_id)
        log.annotate(ep, value="v", incident_id="INC-1")
        self.assertEqual(ep.value, "v")
        self.assertEqual(ep.incident_id, "INC-1")
        log.annotate(ep, incident_id="")  # cleared back to the single absence sentinel
        self.assertIsNone(ep.incident_id)
        self.assertEqual(ep.value, "v")  # untouched — annotate only writes what it is given

    def test_transition_is_the_only_writer_of_status_and_t_end(self):
        """Static guard: no code path outside transition() assigns .status or .t_end. If a future
        edit reintroduces a stray writer, this fails and the invariant is protected."""
        module_file = Path(EpisodeLog.__init__.__globals__["__file__"])
        text = module_file.read_text(encoding="utf-8")
        # Real attribute ASSIGNMENTS only ( `foo.status = ...` ), never comparisons ( `== ` ).
        writers = re.findall(r"\b\w+\.(status|t_end)\s*=(?!=)", text)
        self.assertEqual(
            sorted(writers), ["status", "t_end"],
            f"exactly one status writer and one t_end writer expected, found {writers}",
        )
        # ...and both of them live inside transition().
        transition_body = text.split("def transition(")[1].split("\n    def ")[0]
        self.assertIn("episode.status = status", transition_body)
        self.assertIn("episode.t_end = t if", transition_body)


if __name__ == "__main__":
    unittest.main()
