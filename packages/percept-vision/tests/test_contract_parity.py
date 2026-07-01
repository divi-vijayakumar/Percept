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


if __name__ == "__main__":
    unittest.main()
