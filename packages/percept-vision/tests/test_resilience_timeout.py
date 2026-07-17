"""with_retry timeout_s — a wedged provider call must ABORT, never hang the run.

Regression for the observed 3h `sock_send` hang: a provider's own request timeout does not always
bound a stuck socket send, so with_retry gets a hard per-attempt asyncio.wait_for bound.
"""
from __future__ import annotations

import asyncio
import time
import unittest

from percept.backends._resilience import with_retry


class WithRetryTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_hanging_call_aborts_within_bound(self):
        async def hang():
            await asyncio.sleep(100)  # simulate a wedged socket send
        t0 = time.monotonic()
        with self.assertRaises((asyncio.TimeoutError, Exception)):
            # 3 attempts * 0.05s timeout + tiny backoff — must finish in << 100s, i.e. it did NOT hang
            await with_retry(hang, attempts=3, base_delay_s=0.0, timeout_s=0.05)
        self.assertLess(time.monotonic() - t0, 5.0, "with_retry hung instead of timing out")

    async def test_timeout_retries_then_a_success_returns(self):
        calls = {"n": 0}
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                await asyncio.sleep(100)  # first two attempts wedge
            return "ok"
        out = await with_retry(flaky, attempts=3, base_delay_s=0.0, timeout_s=0.05)
        self.assertEqual(out, "ok")
        self.assertEqual(calls["n"], 3)

    async def test_no_timeout_is_unchanged(self):
        async def quick():
            return 42
        self.assertEqual(await with_retry(quick, timeout_s=None), 42)
        self.assertEqual(await with_retry(quick), 42)  # default: no bound, byte-for-byte


if __name__ == "__main__":
    unittest.main()
