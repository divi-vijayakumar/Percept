"""D6(b) — the documented bring-your-own-admission example actually runs, and stays honest.

`examples/byo_admission_detector.py` is the runnable proof behind docs/admission/detectors.md's
"Admission is not extraction" section: `register_detector` + the tier-0 adapters decide WHEN to look
closely; neither carries an extracted fact (`invoice_number = "INV-2291"`) into the judge. A client went
looking for the OCR seam once and found a dead one — this test keeps the live one from rotting.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_EXAMPLE = (
    Path(__file__).resolve().parents[3] / "examples" / "byo_admission_detector.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("byo_admission_detector", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class ByoAdmissionExampleTests(unittest.TestCase):
    def test_the_example_file_is_where_the_docs_say_it_is(self):
        self.assertTrue(_EXAMPLE.exists(), f"missing example: {_EXAMPLE}")

    def test_both_seams_fire_on_the_page_turn_and_only_there(self):
        out = _load().demo()
        # The fixture turns the page at t=2.0 and again at t=4.0; frames 0/1/3 are unchanged text.
        self.assertEqual(out["admitted_ts"], [2.0, 4.0])  # seam 1: tier-0 adapter -> judge admission
        self.assertEqual(out["boundary_ts"], [2.0, 4.0])  # seam 2: registered detector -> boundaries

    def test_the_seams_carry_a_time_not_a_reading(self):
        """The framing the docs must not overstate: admission is not extraction."""
        out = _load().demo()
        self.assertTrue(all(isinstance(t, float) for t in out["admitted_ts"]))
        self.assertTrue(all(isinstance(t, float) for t in out["boundary_ts"]))

    def test_the_example_leaves_the_registry_clean(self):
        from percept.edge.detectors import registered

        _load().demo()
        self.assertNotIn("document-changed", registered())


if __name__ == "__main__":
    unittest.main()
