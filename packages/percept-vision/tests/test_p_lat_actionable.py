"""PROBE SCAFFOLD — P-LAT-ACTIONABLE-1 (Viability §7, board card A27, gate G3-B).

**Scaffold only — gated on bars-locked, no for-record run.** This proves the WIRING a future
for-record P-LAT-ACTIONABLE-1 run needs, not a real latency verdict about Percept itself.

**What G3-B measures, and why it exists alongside G3-A.** ``P-LAT-RUNTIME-1`` (card A26, G3-A)
measures ``admitted_ts -> decision_emitted_ts`` (``JudgmentTiming.admitted_to_decision_s``) — the
judge-funnel leg ONLY. It is blind to how long tier-0 took to *admit* the episode in the first
place. This probe measures the ACTIONABLE latency instead: ``decision_eligible_ts ->
decision_emitted_ts``, where ``decision_eligible_ts`` is the labeled/synthetic-eligibility instant
(``percept.contracts.GroundTruth.decision_eligible_ts``, by construction — cards A16/A6/A27's own
``eval/viability/ground_truth.py``) — i.e. the first instant a competent human (or the fixture's own
deterministic rubric) could confidently judge the condition. Actionable latency therefore INCLUDES
admission delay (how long tier-0 took to catch up to eligibility and admit) + evidence assembly, on
top of the judge-funnel leg G3-A already covers. A slow tier-0 that admits well after eligibility is
INVISIBLE to G3-A (which only starts its clock at ``admitted_ts``) but is caught by G3-B. Per
Viability §7, Reflex requires BOTH G3-A (< 2s) AND G3-B (< 2s) — a fast judge funnel is not enough
if tier-0 itself is slow to admit.

**Reuse only — no source edits.** This file imports, unmodified:
  * ``eval/viability/fixtures/core_archetypes.py`` (``build_core_archetype_fixtures``) — the six
    CORE gating archetypes' deterministic clips + by-construction :class:`GroundTruth`, card A16.
  * ``eval/viability/ground_truth.py`` (``CORE_ARCHETYPE_ELIGIBLE_AFTER_S``) — the documented
    per-archetype eligibility offsets those fixtures' ``GroundTruth.decision_eligible_ts`` values
    are built from (cards A6/A16), used here only to hand-verify expected numbers, never
    recomputed independently.
  * ``percept.contracts.JudgmentTiming`` — the production timing record (card A5).
  * ``percept.harness.metrics.MetricsCollector`` / ``PercentileSummary`` (card A13).
  * ``percept.clock.FakeClock`` (card A1).

``eval/viability`` is not (yet) an installed package (same as
``eval/viability/fixtures/test_core_archetypes.py``) — this file makes its directory importable the
same way, so ``from core_archetypes import ...`` / ``from ground_truth import ...`` work regardless
of how pytest discovers/imports this file.

**How ACTIONABLE latency is represented without a new source field.** ``JudgmentTiming`` has no
``decision_eligible_ts`` field (by hard invariant — see :class:`GroundTruth`'s docstring: eligibility
is an EVALUATION-ONLY concept, never a field on a live/production timing record). Rather than
reimplementing percentile math or editing ``MetricsCollector``, each synthetic timing here sets
``ingest_observed_ts := decision_eligible_ts`` (the GroundTruth value) instead of a "local receipt of
evidence" instant. This is a legitimate reuse of ``JudgmentTiming``'s own existing
``ingest_to_decision_s()`` helper — it already computes exactly
``decision_emitted_ts - ingest_observed_ts`` — so ``MetricsCollector.latency_summary().
ingest_to_decision`` becomes the ACTIONABLE (G3-B) p95 for free, through the real unmodified
aggregation path, while ``.admitted_to_decision`` (unaffected by this substitution) remains the
RUNTIME (G3-A) p95 on the very same ingested samples. ``JudgmentTiming``'s monotonicity invariant
(``ingest_observed_ts <= admitted_ts``) then structurally enforces the correct reading of "admission
delay": ``admitted_ts`` may never precede ``decision_eligible_ts`` in this scaffold, i.e. tier-0
cannot admit an episode before it is even eligible — it can only be prompt (``admitted_ts ==
decision_eligible_ts``) or late (``admitted_ts > decision_eligible_ts``), which is exactly the
admission-delay quantity G3-B exists to catch.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pytest

# Make `eval/viability/fixtures/` importable as bare modules, same pattern as
# `eval/viability/fixtures/test_core_archetypes.py` (eval/viability is not an installed package).
_EVAL_VIABILITY_FIXTURES = Path(__file__).resolve().parents[3] / "eval" / "viability" / "fixtures"
_EVAL_VIABILITY = Path(__file__).resolve().parents[3] / "eval" / "viability"
sys.path.insert(0, str(_EVAL_VIABILITY_FIXTURES))
sys.path.insert(0, str(_EVAL_VIABILITY))

# `eval/viability/` is intentionally git-untracked and gitignored (see commit cc46b37,
# "untrack internal product-strategy docs before any public push") — it holds the founder
# bars / GO-NOGO memo / quality-gate protocol and is kept LOCAL-ONLY on machines that have run
# the viability harness, never shipped or checked into the public repo. On a clean checkout (and
# in CI) `core_archetypes.py` / `ground_truth.py` genuinely do not exist, so this whole scaffold
# — which reuses those fixtures verbatim, by design (see module docstring) — is unimportable and
# must skip cleanly rather than error at collection. When the fixtures ARE present (e.g. locally,
# with the viability harness checked out alongside the repo), the tests run for real and assert
# exactly what they always did.
pytest.importorskip(
    "core_archetypes",
    reason="eval/viability/fixtures/core_archetypes.py is git-untracked/gitignored (internal "
    "viability harness, kept local-only) — not present on a clean checkout/CI",
)
pytest.importorskip(
    "ground_truth",
    reason="eval/viability/ground_truth.py is git-untracked/gitignored (internal viability "
    "harness, kept local-only) — not present on a clean checkout/CI",
)

from core_archetypes import build_core_archetype_fixtures  # noqa: E402
from ground_truth import CORE_ARCHETYPE_ELIGIBLE_AFTER_S  # noqa: E402

from percept.clock import FakeClock
from percept.contracts import JudgmentTiming
from percept.harness.metrics import MetricsCollector, PercentileSummary, percentile

# ---------------------------------------------------------------------------
# Combined Reflex gate — Viability §7: BOTH G3-A and G3-B must be < 2s.
# ---------------------------------------------------------------------------

REFLEX_THRESHOLD_S = 2.0


def reflex_eligible(runtime_p95_s: float, actionable_p95_s: float) -> bool:
    """Viability §7 combined Reflex gate: G3-A (runtime, admitted->decision) AND G3-B (actionable,
    decision_eligible->decision_emitted) must BOTH be strictly under 2s."""
    return runtime_p95_s < REFLEX_THRESHOLD_S and actionable_p95_s < REFLEX_THRESHOLD_S


def _timing(
    *,
    decision_eligible_ts: float,
    admitted_ts: float,
    judge_wait_s: float = 0.0,
    judge_duration_s: float = 0.0,
    decision_overhead_s: float = 0.0,
) -> JudgmentTiming:
    """Build one synthetic :class:`JudgmentTiming` where ``ingest_observed_ts`` is repurposed to
    carry ``decision_eligible_ts`` (see module docstring) so ``ingest_to_decision_s()`` reads as the
    ACTIONABLE (G3-B) latency and ``admitted_to_decision_s()`` reads as the RUNTIME (G3-A) latency,
    both through the real unmodified ``JudgmentTiming``/``MetricsCollector`` path."""
    judge_started_ts = admitted_ts + judge_wait_s
    judge_completed_ts = judge_started_ts + judge_duration_s
    decision_emitted_ts = judge_completed_ts + decision_overhead_s
    return JudgmentTiming(
        ingest_observed_ts=decision_eligible_ts,
        admitted_ts=admitted_ts,
        judge_started_ts=judge_started_ts,
        judge_completed_ts=judge_completed_ts,
        decision_emitted_ts=decision_emitted_ts,
    )


class TestActionableLatencyHandVerifiedSyntheticSet(unittest.TestCase):
    """P-LAT-ACTIONABLE-1's core property: feed a synthetic set with a KNOWN,
    hand-verifiable answer through MetricsCollector and confirm the p95 it recovers for the
    ACTIONABLE (decision_eligible -> decision_emitted) latency is exactly right."""

    def test_actionable_p95_recovered_exactly_on_known_set(self):
        # Five fixtures, each with a distinct decision_eligible_ts and a distinct total actionable
        # latency (decision_emitted_ts - decision_eligible_ts), hand-chosen so the answer is exact.
        # actionable latencies: 0.5, 1.0, 1.2, 1.5, 1.9 (same shape as P-LAT-RUNTIME-1's GREEN set)
        specs = [
            (0.0, 0.5),
            (2.0, 1.0),
            (5.0, 1.2),
            (10.0, 1.5),
            (20.0, 1.9),
        ]
        collector = MetricsCollector(FakeClock())
        for decision_eligible_ts, actionable_s in specs:
            collector.record_judgment(
                _timing(
                    decision_eligible_ts=decision_eligible_ts,
                    admitted_ts=decision_eligible_ts,  # prompt admission -> isolates the number
                    decision_overhead_s=actionable_s,
                )
            )

        report = collector.latency_summary()
        # nearest-rank p95 of n=5 values (rank = ceil(0.95*5) = 5, 1-indexed) is the max: 1.9.
        expected = percentile([a for _, a in specs], 95)
        self.assertAlmostEqual(expected, 1.9)
        self.assertAlmostEqual(report.ingest_to_decision.p95, expected)
        self.assertEqual(report.ingest_to_decision.n, 5)

    def test_combined_reflex_gate_on_synthetic_inputs(self):
        """The Viability §7 combined gate: BOTH G3-A and G3-B must be < 2s for Reflex."""
        # both comfortably under 2s -> Reflex eligible
        self.assertTrue(reflex_eligible(runtime_p95_s=0.4, actionable_p95_s=1.1))
        # boundary: exactly 2.0s on either side fails (strict "< 2s", matching P-LAT-RUNTIME-1's
        # GREEN-band convention)
        self.assertFalse(reflex_eligible(runtime_p95_s=2.0, actionable_p95_s=0.5))
        self.assertFalse(reflex_eligible(runtime_p95_s=0.5, actionable_p95_s=2.0))
        # both over -> not eligible
        self.assertFalse(reflex_eligible(runtime_p95_s=2.5, actionable_p95_s=3.0))
        # runtime fine but actionable blows the budget (the G3-B point) -> not eligible
        self.assertFalse(reflex_eligible(runtime_p95_s=0.3, actionable_p95_s=2.4))


class TestLateAdmissionDemonstration(unittest.TestCase):
    """THE G3-B point, made explicit: a run whose RUNTIME latency (admitted->decision) is small but
    whose ACTIONABLE latency (decision_eligible->decision_emitted) is large because tier-0 admitted
    late. G3-A alone would call this run GREEN/Reflex-eligible; G3-B (and the combined gate) must
    not."""

    def test_small_runtime_large_actionable_due_to_late_admission(self):
        # Every sample: tier-0 admits 2.2s AFTER the episode was already eligible (a slow tier-0),
        # then the judge funnel itself is fast (0.3s admitted->decision) once admission finally
        # happens. G3-A only starts its clock at admitted_ts, so it never sees the 2.2s the episode
        # spent eligible-but-not-yet-admitted.
        LATE_ADMISSION_DELAY_S = 2.2
        FAST_RUNTIME_S = 0.3

        collector = MetricsCollector(FakeClock())
        eligible_ts_values = [0.0, 10.0, 20.0, 30.0, 40.0]
        for decision_eligible_ts in eligible_ts_values:
            admitted_ts = decision_eligible_ts + LATE_ADMISSION_DELAY_S
            collector.record_judgment(
                _timing(
                    decision_eligible_ts=decision_eligible_ts,
                    admitted_ts=admitted_ts,
                    decision_overhead_s=FAST_RUNTIME_S,
                )
            )

        report = collector.latency_summary()
        runtime_p95 = report.admitted_to_decision.p95  # G3-A
        actionable_p95 = report.ingest_to_decision.p95  # G3-B

        # Hand-verifiable: every sample is identical, so p95 == the single value in each case.
        self.assertAlmostEqual(runtime_p95, FAST_RUNTIME_S)
        self.assertAlmostEqual(actionable_p95, LATE_ADMISSION_DELAY_S + FAST_RUNTIME_S)

        # The whole point: the two numbers DIFFER, by exactly the admission delay.
        self.assertNotEqual(runtime_p95, actionable_p95)
        self.assertAlmostEqual(actionable_p95 - runtime_p95, LATE_ADMISSION_DELAY_S)

        # G3-A alone: this judge funnel leg is GREEN/Reflex-fast in isolation.
        self.assertLess(runtime_p95, REFLEX_THRESHOLD_S)
        # G3-B: the same run is NOT actionable-fast — the slow tier-0 admission blows the budget.
        self.assertGreaterEqual(actionable_p95, REFLEX_THRESHOLD_S)

        # Combined Reflex gate must catch what G3-A alone would miss.
        self.assertFalse(reflex_eligible(runtime_p95, actionable_p95))

    def test_prompt_admission_makes_the_two_numbers_converge(self):
        """Contrast case: when tier-0 admits PROMPTLY (admitted_ts == decision_eligible_ts, zero
        admission delay), RUNTIME and ACTIONABLE latency are identical — G3-A and G3-B agree, and
        neither hides anything from the other. Confirms the divergence above is caused by the
        admission delay specifically, not some other artifact of the two helpers."""
        collector = MetricsCollector(FakeClock())
        for decision_eligible_ts in (0.0, 5.0, 15.0):
            collector.record_judgment(
                _timing(
                    decision_eligible_ts=decision_eligible_ts,
                    admitted_ts=decision_eligible_ts,  # prompt: no admission delay
                    decision_overhead_s=0.4,
                )
            )
        report = collector.latency_summary()
        self.assertAlmostEqual(report.admitted_to_decision.p95, report.ingest_to_decision.p95)

    def test_actionable_latency_never_less_than_runtime_latency(self):
        """Structural invariant this scaffold relies on: since admission can never precede
        eligibility (JudgmentTiming's own monotonicity invariant, with ingest_observed_ts standing
        in for decision_eligible_ts here), ACTIONABLE latency is always >= RUNTIME latency, across a
        spread of admission delays including zero."""
        collector = MetricsCollector(FakeClock())
        for i, delay in enumerate([0.0, 0.1, 0.75, 1.5, 3.0]):
            collector.record_judgment(
                _timing(
                    decision_eligible_ts=float(i) * 100.0,
                    admitted_ts=float(i) * 100.0 + delay,
                    decision_overhead_s=0.2,
                )
            )
        for timing in collector._latency_samples:  # noqa: SLF001 (test-only introspection)
            self.assertGreaterEqual(timing.ingest_to_decision_s(), timing.admitted_to_decision_s())


class TestActionableLatencyOverCoreArchetypeFixtures(unittest.TestCase):
    """Exercises the SAME reuse (core_archetypes.py + ground_truth.py's by-construction GroundTruth)
    the card names, over the real neutral fixture set (card A16) rather than hand-authored numbers
    alone — proving the wiring works against the actual fixtures a for-record run will use."""

    def test_actionable_latency_matches_hand_computed_ground_truth_offset(self):
        fixtures = build_core_archetype_fixtures()
        positive = [f for f in fixtures if f.positive]
        self.assertGreater(len(positive), 0)

        # Fixed, tiny synthetic funnel: prompt admission, negligible judge/decision overhead — so
        # each fixture's actionable latency is (hand-verifiably) just that overhead, letting this
        # test confirm decision_eligible_ts round-trips correctly from GroundTruth through
        # MetricsCollector without being swamped by an arbitrarily-chosen funnel shape.
        FUNNEL_OVERHEAD_S = 0.2

        collector = MetricsCollector(FakeClock())
        expected_eligible_ts: dict[str, float] = {}
        for fixture in positive:
            gt = fixture.ground_truth
            assert gt is not None  # positive fixtures always carry a GroundTruth (A16 invariant)
            expected_eligible_ts[fixture.fixture_id] = gt.decision_eligible_ts
            # Cross-check against the documented per-archetype offset table this GroundTruth was
            # built from, for the archetypes that use the plain default (not persistence's explicit
            # override) — hand-verifiable, not a tautology against GroundTruth itself.
            if fixture.archetype in ("state", "transition", "brief_action", "absence"):
                event_start_ts = fixture.meta.get("event_start_ts")
                if event_start_ts is not None:
                    self.assertAlmostEqual(
                        gt.decision_eligible_ts,
                        event_start_ts + CORE_ARCHETYPE_ELIGIBLE_AFTER_S[fixture.archetype],
                    )
            collector.record_judgment(
                _timing(
                    decision_eligible_ts=gt.decision_eligible_ts,
                    admitted_ts=gt.decision_eligible_ts,
                    decision_overhead_s=FUNNEL_OVERHEAD_S,
                )
            )

        report = collector.latency_summary()
        # Every sample's actionable latency is exactly FUNNEL_OVERHEAD_S (prompt admission,
        # negligible funnel) regardless of how spread out the underlying decision_eligible_ts values
        # are -> p50/p95/p99 all collapse to the same hand-verifiable constant.
        self.assertAlmostEqual(report.ingest_to_decision.p50, FUNNEL_OVERHEAD_S)
        self.assertAlmostEqual(report.ingest_to_decision.p95, FUNNEL_OVERHEAD_S)
        self.assertAlmostEqual(report.ingest_to_decision.p99, FUNNEL_OVERHEAD_S)
        self.assertEqual(report.ingest_to_decision.n, len(positive))

    def test_late_admission_over_real_fixtures_blows_the_actionable_budget(self):
        """Same G3-B demonstration as TestLateAdmissionDemonstration, but driven off the real A16
        fixture set's GroundTruth values instead of hand-authored eligibility timestamps."""
        fixtures = build_core_archetype_fixtures()
        positive = [f for f in fixtures if f.positive]

        LATE_ADMISSION_DELAY_S = 2.5
        FAST_RUNTIME_S = 0.25

        collector = MetricsCollector(FakeClock())
        for fixture in positive:
            gt = fixture.ground_truth
            assert gt is not None
            collector.record_judgment(
                _timing(
                    decision_eligible_ts=gt.decision_eligible_ts,
                    admitted_ts=gt.decision_eligible_ts + LATE_ADMISSION_DELAY_S,
                    decision_overhead_s=FAST_RUNTIME_S,
                )
            )

        report = collector.latency_summary()
        runtime_p95 = report.admitted_to_decision.p95
        actionable_p95 = report.ingest_to_decision.p95

        self.assertAlmostEqual(runtime_p95, FAST_RUNTIME_S)
        self.assertAlmostEqual(actionable_p95, LATE_ADMISSION_DELAY_S + FAST_RUNTIME_S)
        self.assertLess(runtime_p95, REFLEX_THRESHOLD_S)
        self.assertGreaterEqual(actionable_p95, REFLEX_THRESHOLD_S)
        self.assertFalse(reflex_eligible(runtime_p95, actionable_p95))


class TestPercentileHelperAgreesWithCollector(unittest.TestCase):
    """Cross-check, mirroring P-LAT-BASE-1/P-LAT-RUNTIME-1's own cross-checks: the bare
    ``percentile()``/``PercentileSummary.of()`` helpers agree with the collector's aggregation on the
    same raw actionable-latency samples."""

    def test_percentile_helper_matches_collector_ingest_to_decision(self):
        raw_actionable_s = [0.3, 0.6, 0.9, 1.4, 2.1, 2.9]
        collector = MetricsCollector(FakeClock())
        for i, actionable_s in enumerate(raw_actionable_s):
            collector.record_judgment(
                _timing(
                    decision_eligible_ts=float(i) * 50.0,
                    admitted_ts=float(i) * 50.0,
                    decision_overhead_s=actionable_s,
                )
            )
        recovered = collector.latency_summary().ingest_to_decision
        expected = PercentileSummary.of(raw_actionable_s)
        # Float tolerance, not exact equality: decision_eligible_ts values here are nonzero
        # (float(i) * 50.0), so decision_emitted_ts - decision_eligible_ts reintroduces ordinary
        # floating-point subtraction error the zero-based synthetic sets elsewhere in this file
        # avoid by construction (see P-LAT-BASE-1's own docstring on this exact tradeoff).
        self.assertAlmostEqual(recovered.p50, expected.p50)
        self.assertAlmostEqual(recovered.p95, expected.p95)
        self.assertAlmostEqual(recovered.p99, expected.p99)
        self.assertEqual(recovered.n, expected.n)
        self.assertAlmostEqual(percentile(raw_actionable_s, 95), recovered.p95)


if __name__ == "__main__":
    unittest.main()
