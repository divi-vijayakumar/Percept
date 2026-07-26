"""The v0.6 SDK voice diet, pinned (the documented regression).

The voice embodiment moved app-side into ``templates/realtime_voice_watch_agent``. This locks the three
promises that make the move real:

  * ``RuntimeSession.attach_transport`` / ``deliver_fire`` are RAISING STUBS (fail loud, point at the
    template) — not silent no-ops, not delegating shims;
  * ``import percept.realtime`` raises an ImportError that names the template;
  * importing the SDK (``percept`` + ``percept.runtime``) drags NO voice vendor (pipecat/fastapi/uvicorn)
    into the import graph — the perception package does not own the voice stack.
"""
from __future__ import annotations

import subprocess
import sys
import unittest

from percept.runtime.session import RuntimeSession

#: voice-stack vendors that must never be pulled by importing the SDK (they live template-side now).
_VOICE_VENDORS = ("pipecat", "fastapi", "uvicorn")


class _FakeCore:
    """A do-nothing core — the stubs must raise before touching it."""


class RaisingStubTests(unittest.TestCase):
    def test_attach_transport_raises_and_points_at_the_template(self):
        session = RuntimeSession(_FakeCore())
        with self.assertRaises(RuntimeError) as ctx:
            session.attach_transport(object())
        msg = str(ctx.exception)
        self.assertIn("attach_transport", msg)
        self.assertIn("templates/realtime_voice_watch_agent", msg)
        self.assertIn("PerceptBridge", msg)

    def test_deliver_fire_raises_and_points_at_the_template(self):
        session = RuntimeSession(_FakeCore())
        with self.assertRaises(RuntimeError) as ctx:
            session.deliver_fire(object())
        msg = str(ctx.exception)
        self.assertIn("deliver_fire", msg)
        self.assertIn("templates/realtime_voice_watch_agent", msg)
        self.assertIn("on_incident", msg)


class RealtimeModuleMovedTests(unittest.TestCase):
    def test_importing_percept_realtime_raises_a_pointed_import_error(self):
        with self.assertRaises(ImportError) as ctx:
            import percept.realtime  # noqa: F401
        msg = str(ctx.exception)
        self.assertIn("percept.realtime", msg)
        self.assertIn("templates/realtime_voice_watch_agent", msg)
        self.assertIn("percept.contracts", msg)   # where the VoiceTransport contract still lives


class NoVoiceVendorAtImportTests(unittest.TestCase):
    def test_importing_the_sdk_pulls_no_voice_vendor(self):
        # Fresh interpreter so another test's import can't pollute sys.modules. `import percept.runtime`
        # covers the session/worker surface where attach_transport used to drag pipecat in.
        code = (
            "import sys\n"
            "import percept, percept.runtime\n"
            f"leaked = [m for m in {_VOICE_VENDORS!r} if m in sys.modules]\n"
            "print(','.join(leaked))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, f"SDK import failed:\n{out.stderr}")
        leaked = out.stdout.strip()
        self.assertEqual(leaked, "", f"importing the SDK leaked a voice vendor: {leaked}")


if __name__ == "__main__":
    unittest.main()
