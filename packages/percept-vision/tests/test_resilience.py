"""Circuit-breaker telemetry: state + opens counter exposed via snapshot()."""

from __future__ import annotations

import unittest

from percept.backends._resilience import CircuitBreaker


class CircuitBreakerTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_tracks_state_failures_and_opens(self):
        b = CircuitBreaker(fail_threshold=2, reset_after_s=999.0)
        self.assertEqual(b.snapshot(), {"state": "closed", "failures": 0, "opens": 0})

        b.record_failure()
        self.assertEqual(b.state, "closed")  # below threshold
        b.record_failure()  # trips closed -> open
        snap = b.snapshot()
        self.assertEqual(snap["state"], "open")
        self.assertEqual(snap["opens"], 1)
        self.assertFalse(b.allow())  # open short-circuits

        b.record_success()  # recovery
        self.assertEqual(b.snapshot(), {"state": "closed", "failures": 0, "opens": 1})

    async def test_opens_counts_distinct_trips_not_every_failure(self):
        b = CircuitBreaker(fail_threshold=1, reset_after_s=0.0)
        b.record_failure()  # opens (1)
        self.assertEqual(b.snapshot()["opens"], 1)
        self.assertTrue(b.allow())  # reset_after_s=0 -> half-open immediately
        b.record_failure()  # opens again (2)
        self.assertEqual(b.snapshot()["opens"], 2)


if __name__ == "__main__":
    unittest.main()
