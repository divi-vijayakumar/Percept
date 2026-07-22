"""L1: episodes retain evidence HANDLES, never decoded pixels (card Y14).

Z1 proved a count-bounded buffer is not byte-bounded: the SAME ``max_episodes`` costs kilobytes of
8x8 test frames and hundreds of MB of 1080p frames. ``max_episodes`` bounds the COUNT of episodes,
not their BYTES — so an episode that pins a decoded frame makes the log's memory scale with source
resolution and, unchecked, with the number of observations. These tests assert the fix: a decoded
frame is replaced by a small :class:`~percept.episodes.EvidenceHandle` before storage, cheap
descriptors pass through byte-for-byte, and — the soak assertion — retained decoded bytes do NOT grow
with N beyond the declared count bound.
"""

import unittest

from percept.agent import Percept
from percept.episodes import EvidenceHandle, EpisodeLog, evidence_handle
from percept.events import frame_nbytes

# ALWAYS_NEW: every observation opens a fresh episode (exercises the OPEN store path, and lets us
# drive the count bound). The judge never inspects the frame here — it is not asserting intelligence.
ALWAYS_NEW = lambda summaries, exemplar: None  # noqa: E731
ALWAYS_CONTINUE = lambda summaries, exemplar: summaries[-1]["id"] if summaries else None  # noqa: E731

# A decoded frame the harness can measure: raw bytes, so ``frame_nbytes`` reports its true size the
# way it does for a numpy buffer. 640x480 single-channel ~= 300 KB, a realistic per-frame cost.
_FRAME_W, _FRAME_H = 640, 480
_FRAME_BYTES = _FRAME_W * _FRAME_H


def _frame(seed: int) -> bytes:
    """A distinct, MEASURABLE decoded frame (~300 KB) — the thing that must not be retained."""
    return bytes([seed % 251]) * _FRAME_BYTES


class TestEvidenceHandleConversion(unittest.TestCase):
    def test_decoded_frame_is_stored_as_a_handle_not_pixels(self):
        log = EpisodeLog("g")
        ep, _ = log.observe(1.0, _frame(1), ALWAYS_NEW)
        self.assertIsInstance(ep.exemplar, EvidenceHandle)
        self.assertIsInstance(ep.representative, EvidenceHandle)
        # the frame's pixels are NOT retained — the handle measures ~0 under the same Z1 function
        self.assertEqual(frame_nbytes(ep.exemplar), 0)
        # ...but the size it stands in for is recorded, so the saving is observable
        self.assertEqual(ep.exemplar.source_nbytes, _FRAME_BYTES)

    def test_cheap_descriptors_pass_through_byte_for_byte(self):
        log = EpisodeLog("g")
        ep_tok, _ = log.observe(1.0, "cup", ALWAYS_NEW)
        ep_dict, _ = log.observe(2.0, {"total": 3}, ALWAYS_NEW)
        # a subject token and a small dict are cheap — retained AS-IS (pre-Y14 behaviour)
        self.assertEqual(ep_tok.exemplar, "cup")
        self.assertEqual(ep_dict.exemplar, {"total": 3})
        self.assertNotIsInstance(ep_tok.exemplar, EvidenceHandle)

    def test_fingerprint_is_populated_for_a_griddable_frame(self):
        # the handle carries a cheap pixel-free signature the judge can compare on
        h = evidence_handle(_frame(7))
        self.assertIsInstance(h, EvidenceHandle)
        self.assertIsNotNone(h.fingerprint)

    def test_conversion_is_idempotent(self):
        h = evidence_handle(_frame(1))
        self.assertIs(evidence_handle(h), h)

    def test_continuation_representative_is_also_a_handle(self):
        # the LIVE representative (updated on every continuation) must be a handle too, not pixels
        log = EpisodeLog("g")
        log.observe(1.0, _frame(1), ALWAYS_CONTINUE)
        ep, is_new = log.observe(2.0, _frame(2), ALWAYS_CONTINUE)
        self.assertFalse(is_new)
        self.assertIsInstance(ep.representative, EvidenceHandle)
        self.assertEqual(frame_nbytes(ep.representative), 0)

    def test_judge_hot_path_never_sees_retained_pixels(self):
        # summaries() is what the judge compares the incoming observation AGAINST; it must carry no
        # pixels even after decoded frames have been stored (the retention side is pixel-free).
        log = EpisodeLog("g")
        for i in range(3):
            log.observe(float(i), _frame(i), ALWAYS_NEW)
        for s in log.summaries():
            self.assertNotIn("exemplar", s)
            self.assertNotIn("representative", s)


class TestEvidenceSoak(unittest.TestCase):
    def test_retained_bytes_do_not_grow_with_N(self):
        """SOAK: over N observations retained decoded bytes stay bounded by COUNT, not by N.

        Every observation opens a NEW episode (ALWAYS_NEW) carrying a ~300 KB frame. Under the naive
        'store the frame' design retained bytes would climb to ``max_episodes x 2 x frame_bytes`` and
        the log would pin tens of MB; under Y14 handles it pins ZERO decoded bytes at every N.
        """
        log = EpisodeLog("g", max_episodes=32)
        checkpoints: dict[int, int] = {}
        for i in range(1500):
            log.observe(float(i), _frame(i), ALWAYS_NEW)
            n = i + 1
            if n in (50, 200, 800, 1500):
                checkpoints[n] = log.retained_evidence_nbytes()

        # the log is bounded in COUNT, as ever
        self.assertEqual(log.count(), 32)
        # ...and pins ZERO decoded pixels at every checkpoint — independent of N
        for n, retained in checkpoints.items():
            self.assertEqual(retained, 0, f"pinned {retained} decoded bytes at N={n}")
        # bytes are not merely small but FLAT across N (the soak invariant)
        self.assertEqual(len(set(checkpoints.values())), 1)

        # the saving is real, not an artefact of tiny frames: the naive per-episode-frame design
        # would have pinned the plateau below (~19 MB here), which scales with source RESOLUTION.
        naive_plateau = log.max_episodes * 2 * _FRAME_BYTES
        self.assertGreater(naive_plateau, 15_000_000)

    def test_count_event_path_stores_handles_end_to_end(self):
        # the defect was in the counting path (Percept.count_event); prove it stores handles now.
        a = Percept.create(discover_plugins=False)
        ep, is_new = a.count_event("cups", _frame(1), ALWAYS_NEW, t=1.0)
        self.assertTrue(is_new)
        self.assertIsInstance(ep.exemplar, EvidenceHandle)
        self.assertEqual(frame_nbytes(ep.exemplar), 0)


if __name__ == "__main__":
    unittest.main()
