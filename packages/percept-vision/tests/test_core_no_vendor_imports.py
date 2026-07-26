"""Invariant 7 as a GATE (v3 §11): importing core cognition must pull NO vendor SDK.

The "model-agnostic core" promise is only real if the cognition modules physically cannot drag a vendor SDK
into the import graph. This locks it: in a FRESH interpreter (so another test's backend import can't pollute
sys.modules), import only the core and assert no vendor module appeared. Backends lazy-import their SDK
inside methods, so this stays true — and pre-stages the percept-core package split."""
import subprocess
import sys
import unittest

# the run-time + plan-time cognition surface that must stay vendor-free
_CORE = (
    "percept.gate", "percept.agent", "percept.behavior", "percept.concern", "percept.sensorplan",
    "percept.individuation", "percept.rhythm", "percept.measurement", "percept.scheduler",
    "percept.observability", "percept.escalation", "percept.clock", "percept.goals", "percept.entity",
    "percept.contracts",  # the folded-in wire-contract membrane — must stay pure stdlib
)
_VENDOR = ("google.genai", "anthropic", "openai", "deepgram")


class CoreHasNoVendorImportsTests(unittest.TestCase):
    def test_importing_core_pulls_no_vendor_sdk(self):
        code = (
            "import sys\n"
            f"import {', '.join(_CORE)}\n"
            f"leaked = [m for m in {_VENDOR!r} if m in sys.modules]\n"
            "print(','.join(leaked))\n"
        )
        # fresh interpreter; inherit env (PYTHONPATH) + cwd from the test run so `import percept` resolves
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, f"core import failed:\n{out.stderr}")
        leaked = out.stdout.strip()
        self.assertEqual(leaked, "", f"core cognition leaked vendor imports: {leaked}")


if __name__ == "__main__":
    unittest.main()
