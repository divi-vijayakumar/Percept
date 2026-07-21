"""JS ↔ Python wire-contract version parity.

The Python and JS edges version the wire contract independently. Their convention (documented in
``percept-edge/src/contract.js``) is **same MAJOR == wire-compatible**: a minor bump is an additive,
back-compatible field (e.g. Python 1.2 added ``Verdict.running_answer``, which the JS edge does not
consume), so 1.1 ↔ 1.2 is fine. A **major** divergence, however, means an incompatible wire and must
be deliberate — this test fails on it so it can never slip in silently.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from percept.contracts import CONTRACT_VERSION as PY_VERSION


def _contract_js() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "packages" / "percept-edge" / "src" / "contract.js"
        if candidate.is_file():
            return candidate
    raise unittest.SkipTest("percept-edge/src/contract.js not found (Python-only checkout)")


def _js_contract_version() -> str:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "packages" / "percept-edge" / "src" / "contract.js"
        if candidate.is_file():
            m = re.search(r'CONTRACT_VERSION\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', candidate.read_text())
            if m:
                return m.group(1)
            raise AssertionError(f"CONTRACT_VERSION not found in {candidate}")
    raise unittest.SkipTest("percept-edge/src/contract.js not found (Python-only checkout)")


class ContractVersionParityTests(unittest.TestCase):
    def test_js_and_python_share_major(self) -> None:
        js = _js_contract_version()
        py_major = PY_VERSION.split(".")[0]
        js_major = js.split(".")[0]
        self.assertEqual(
            py_major,
            js_major,
            f"wire-contract MAJOR diverged: Python {PY_VERSION} vs JS {js}. Same major == "
            "wire-compatible; a major bump is a breaking wire change and must be deliberate "
            "(mirror it on both edges).",
        )

class IncidentTwinParityTests(unittest.TestCase):
    """D4: ``contract.js`` is an explicit twin of ``percept.contracts.Incident``. A field added on the
    Python side and not on the JS side is a silent parity break — this catches it at the source level."""

    def test_js_incident_twin_declares_entity_id(self) -> None:
        src = _contract_js().read_text()
        make = src.split("export function makeIncident(", 1)[1].split("export function parseIncident", 1)[0]
        parse = src.split("export function parseIncident(", 1)[1]
        self.assertIn("entity_id = null", make, "makeIncident must default entity_id to null")
        self.assertIn("entity_id", parse, "parseIncident must carry entity_id off the wire")

    def test_python_incident_fields_are_all_named_in_the_js_twin(self) -> None:
        import dataclasses

        from percept.contracts import Incident

        src = _contract_js().read_text()
        twin = src.split("// Incident  {", 1)[1].split("export function parseIncidentSnippet", 1)[0]
        for f in dataclasses.fields(Incident):
            if f.name in {"message", "label_url"}:
                continue  # known JS-side gaps predating D4; documented, not introduced here
            self.assertIn(f.name, twin, f"Incident.{f.name} has no JS twin in contract.js")


class SubjectRefTwinParityTests(unittest.TestCase):
    """Y5: ``contract.js`` is an explicit twin of ``percept.contracts.SubjectRef``. A field added on
    the Python side and not mirrored in the JS ``makeSubjectRef`` is a silent parity break."""

    def test_js_subjectref_twin_declares_every_python_field(self) -> None:
        import dataclasses

        from percept.contracts import SubjectRef

        src = _contract_js().read_text()
        self.assertIn("export function makeSubjectRef(", src, "contract.js must declare makeSubjectRef")
        make = src.split("export function makeSubjectRef(", 1)[1].split("export function parseSubjectRef", 1)[0]
        for f in dataclasses.fields(SubjectRef):
            self.assertIn(f.name, make, f"SubjectRef.{f.name} has no JS twin in makeSubjectRef")


if __name__ == "__main__":
    unittest.main()
