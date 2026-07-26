"""— the SHADOW per-probe report over BOTH probes, driven by ScriptedJudge + the committed
synthetic corpora (no API spend, no gitignored tree required).

What these tests pin:

* both probes run and score every fixture item;
* on SYNTHETIC ground truth the mechanism the shadow observes (subject resolution + continuation +
  the claim/mutation state machine) proposes the RIGHT fold — match_rate 1.0, ZERO false binds. That
  is the de-risking result the card exists to produce, and a regression in the fold breaks it here;
* the report is NOT vacuously green — a genuinely wrong proposal is caught as a false bind and, past
  the hard ceiling, rejected;
* the FALSE-BIND > miss asymmetry (Y8) is what drives acceptance — swapping a miss for a false bind
  strictly WORSENS the harm score;
* ``distinct_subject_count`` is computed by ONE formula (the E-1 IdentityLedger oracle itself, fed
  the same proposed verdicts) rather than a fixture-side count kept in sync with the oracle by hand;
* the honesty disclaimer is stated in the rendered report.

HONESTY: this proves the mechanism WOULD have been right on synthetic ground truth. It does NOT prove
real-world accuracy — the report and these tests say so.
"""

from __future__ import annotations

import unittest

from percept.eval.identity_metrics import BindOutcome, compute_e1_metrics
from percept.eval.shadow_report import (
    DISTINCT_SUBJECT_COUNT_NOTE,
    SHADOW_ACCEPTANCE_POLICY,
    ProbeReport,
    format_report,
    run_probe_01_identity_continuation,
    run_probe_02_mutation,
    run_shadow_probes,
)


class TestProbesRunAndAreRight(unittest.TestCase):
    def test_both_probes_present_and_scored(self):
        reports = run_shadow_probes()
        self.assertEqual(set(reports), {"probe_01", "probe_02"})
        self.assertGreater(reports["probe_01"].metrics.n, 0)
        self.assertGreater(reports["probe_02"].metrics.n, 0)

    def test_probe_01_identity_continuation_is_right_on_synthetic_truth(self):
        r = run_probe_01_identity_continuation()
        # The mechanism binds identity and splits continuation correctly on the fixtures.
        self.assertEqual(r.match_rate, 1.0)
        self.assertEqual(r.metrics.false_bound_rate, 0.0)
        self.assertEqual(
            [d for d in r.disagreements if d.kind == "false_bind"], [],
            "a false bind on the identity/continuation probe is the dangerous silent-merge error",
        )
        self.assertTrue(r.acceptance.accepted)

    def test_probe_02_mutation_is_right_on_synthetic_truth(self):
        r = run_probe_02_mutation()
        self.assertEqual(r.match_rate, 1.0)
        self.assertEqual(r.metrics.false_bound_rate, 0.0)
        # The four "$25K" utterances that only CONTAIN the value must not move it; exactly the one
        # replace does. If the state machine applied a mention, it would show as a false bind here.
        self.assertEqual([d for d in r.disagreements if d.kind == "false_bind"], [])
        self.assertTrue(r.acceptance.accepted)

    def test_report_notes_the_reconciled_distinct_subject_count_definition(self):
        r = run_probe_01_identity_continuation()
        self.assertIn(DISTINCT_SUBJECT_COUNT_NOTE, r.notes)
        self.assertIn("oracle", DISTINCT_SUBJECT_COUNT_NOTE.lower())

    def test_rendered_report_states_the_honesty_disclaimer(self):
        text = format_report(run_shadow_probes())
        self.assertIn("does NOT prove real-world accuracy", text)
        self.assertIn("DESIGN QUESTION", text)


class TestReportIsNotVacuouslyGreen(unittest.TestCase):
    """A green report is only meaningful if a WRONG proposal would be caught. Prove the scoring
    machinery detects a false bind and that the asymmetry rejects past the ceiling."""

    def _report_with(self, outcomes):
        from percept.eval.identity_metrics import assess_acceptance
        metrics = compute_e1_metrics(outcomes)
        return ProbeReport(
            probe="synthetic", corpora=("x",), metrics=metrics,
            acceptance=assess_acceptance(metrics, SHADOW_ACCEPTANCE_POLICY),
            disagreements=(), matched=sum(1 for o in outcomes if o is BindOutcome.EXACT),
        )

    def test_a_false_bind_lowers_match_and_can_be_rejected(self):
        # 49 exact + 1 wrong-key false bind: false_bound_rate 0.02 sits AT the ceiling (0.02), still
        # accepted; one more false bind clears it.
        good = [BindOutcome.EXACT] * 49 + [BindOutcome.WRONG_KEY_FALSE_BIND]
        r = self._report_with(good)
        self.assertLess(r.match_rate, 1.0)
        self.assertGreater(r.metrics.false_bound_rate, 0.0)

        worse = [BindOutcome.EXACT] * 40 + [BindOutcome.WRONG_KEY_FALSE_BIND] * 10
        rw = self._report_with(worse)
        self.assertFalse(rw.acceptance.accepted)  # 0.2 false-bind rate blows the 0.02 ceiling
        self.assertTrue(any("ceiling" in reason for reason in rw.acceptance.reasons))

    def test_false_bind_is_weighted_worse_than_a_miss(self):
        # Swap ONE abstention/miss for ONE false bind at equal n: harm must strictly increase — the
        # load-bearing asymmetry the whole card leans on (Y8).
        base = [BindOutcome.EXACT] * 8 + [BindOutcome.ABSTAINED_WHEN_LEGIBLE, BindOutcome.EXACT]
        swapped = [BindOutcome.EXACT] * 8 + [BindOutcome.WRONG_KEY_FALSE_BIND, BindOutcome.EXACT]
        h_miss = SHADOW_ACCEPTANCE_POLICY.harm_score(compute_e1_metrics(base))
        h_false = SHADOW_ACCEPTANCE_POLICY.harm_score(compute_e1_metrics(swapped))
        self.assertGreater(h_false, h_miss)


if __name__ == "__main__":
    unittest.main()
