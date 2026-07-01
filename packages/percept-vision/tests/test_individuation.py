"""Event-span data association (Build 4) — L1 tests on SYNTHETIC candidates (no pose, no video, no model).

The one property the engine exists to provide: many observations of one underlying event collapse to ONE
event in physical units, so the deduped count is invariant to how densely the stream was observed — and the
SAME engine does it for reps (temporal/source) and steps (descriptor/meaning).
"""

from __future__ import annotations

import unittest

from percept.individuation import EventCandidate, SpanAssociator, associate, descriptor_overlap


def rep(anchor, source="hip", dur=0.4, score=1.0):
    return EventCandidate("rep", anchor - dur / 2, anchor, anchor + dur / 2, source=source, score=score)


def step(anchor, descriptor, source="vlm"):
    return EventCandidate("step", anchor - 0.5, anchor, anchor + 0.5, source=source, descriptor=descriptor)


class RepAssociationTests(unittest.TestCase):
    def test_distinct_reps_beyond_refractory_are_kept(self):
        cands = [rep(t) for t in (0.0, 2.0, 4.0, 6.0)]  # 2s apart, refractory 1.0 -> all distinct
        self.assertEqual(len(associate(cands, refractory_s=1.0)), 4)

    def test_duplicates_inside_refractory_collapse_to_one(self):
        # five near-simultaneous observations of ONE rep -> one event (suppressed dupes)
        cands = [rep(t) for t in (3.00, 3.05, 3.10, 2.95, 3.02)]
        out = associate(cands, refractory_s=1.0)
        self.assertEqual(len(out), 1)
        self.assertGreater(out[0].score, 1.0)  # accumulated confidence from the merged observations

    def test_density_invariance_bounded_duplicate_rate(self):
        # K=6 true reps (1.5s apart). Observe each M times within a tight jitter; the deduped count is K for
        # EVERY density — "denser observation does not create duplicate events" (eval-gate criterion c).
        K, spacing = 6, 1.5
        for m in (1, 2, 4, 8, 16):
            cands = []
            for k in range(K):
                base = k * spacing
                for j in range(m):
                    cands.append(rep(base + 0.01 * j))  # M observations clustered at the true anchor
            self.assertEqual(len(associate(cands, refractory_s=0.7)), K, f"density m={m} changed the count")

    def test_overlapping_spans_merge_widen_the_span(self):
        a = EventCandidate("rep", 0.0, 0.5, 1.0, source="hip", score=1.0)
        b = EventCandidate("rep", 0.8, 1.2, 1.8, source="hip", score=2.0)  # overlaps a, higher score
        out = associate([a, b], refractory_s=0.1)  # refractory tiny: only the span OVERLAP merges them
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0].start_t, out[0].stop_t), (0.0, 1.8))  # widened to cover both
        self.assertEqual(out[0].anchor_t, 1.2)  # higher-score candidate keeps the anchor

    def test_distinct_sources_stay_distinct_unless_cross_source(self):
        cands = [rep(3.0, source="hip"), rep(3.05, source="torso")]
        self.assertEqual(len(associate(cands, refractory_s=1.0)), 2)  # same time, different channel -> 2
        self.assertEqual(len(associate(cands, refractory_s=1.0, cross_source=True)), 1)  # fused as one event


class StepAssociationTests(unittest.TestCase):
    def test_descriptor_paraphrases_dedup_to_one_step(self):
        # the synthesis/process case: paraphrases of one step collapse; a genuinely different step does not.
        cands = [
            step(1.0, "remove the seeds"),
            step(5.0, "removing seeds from the pod"),  # paraphrase of step 1
            step(9.0, "cut the pulp with a knife"),  # a different step
        ]
        out = associate(cands, refractory_s=0.5)  # far apart in time -> only the descriptor dedups them
        self.assertEqual(len(out), 2)

    def test_descriptor_match_beats_pure_time_gap(self):
        self.assertTrue(descriptor_overlap("remove the seeds", "seeds removed"))
        self.assertFalse(descriptor_overlap("remove the seeds", "soak the pulp in water"))


class StreamingAssociatorTests(unittest.TestCase):
    def test_add_reports_new_then_merged(self):
        assoc = SpanAssociator(refractory_s=1.0)
        _e1, new1 = assoc.add(rep(0.0))
        _e2, new2 = assoc.add(rep(0.05))  # duplicate of the first
        _e3, new3 = assoc.add(rep(3.0))  # a distinct rep
        self.assertEqual((new1, new2, new3), (True, False, True))
        self.assertEqual(assoc.count(), 2)

    def test_from_rep_adapts_rhythm_spans(self):
        class _Rep:  # duck-typed Rep
            start_t, peak_t, stop_t, channel, amplitude = 1.0, 1.3, 1.6, "hip", 0.8

        cand = EventCandidate.from_rep(_Rep())
        self.assertEqual((cand.kind, cand.source, cand.anchor_t, cand.amplitude), ("rep", "hip", 1.3, 0.8))


if __name__ == "__main__":
    unittest.main()
