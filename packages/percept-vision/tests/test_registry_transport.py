"""Registry wiring for the `transport` seam — resolves like stt/tts, vendor stays lazy.

The voice-transport seam ships only the `fake` in-core; a real Pipecat/LiveKit transport is a vendor
dep + a WebRTC room to test, so it registers client-side via the `percept.backends` entry point and must
NOT be importable here. Guards the seam kind, `make_transport`, and the no-vendor-import discipline.
"""
from __future__ import annotations

import subprocess
import sys
import unittest

from percept import registry
from percept.backends.fake import FakeVoiceTransport


class TransportSeamTests(unittest.TestCase):
    def test_transport_is_a_seam_kind(self):
        self.assertIn("transport", registry._SEAMS)

    def test_make_transport_resolves_fake(self):
        t = registry.make_transport("fake")
        self.assertIsInstance(t, FakeVoiceTransport)
        self.assertIn("fake", registry.available("transport"))

    def test_unknown_transport_raises_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            registry.make_transport("pipecat")   # not registered in-core (client-side entry point)
        self.assertIn("transport", str(ctx.exception))

    def test_no_vendor_sdk_imported(self):
        # Resolving the seam must not pull a real transport SDK (invariant 7 discipline). Checked in a
        # CLEAN subprocess so an unrelated test that already imported pipecat can't pollute sys.modules.
        code = (
            "import sys; from percept import registry; registry.make_transport('fake');"
            "assert 'pipecat' not in sys.modules and 'livekit' not in sys.modules, 'vendor leaked';"
            "print('CLEAN')"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertIn("CLEAN", out.stdout, msg=out.stderr)


if __name__ == "__main__":
    unittest.main()
