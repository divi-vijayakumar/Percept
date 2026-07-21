"""ADR-001 must not drift back into claiming things the runtime does not do (card X8).

Three claims in ADR-001 were false against the shipped code: §1 named ``GateDecision`` as the gate's
output (it is ``GateOutcome``), §8 described a per-key evidence-sequencing rule with zero production
call sites, and §5 claimed all latency math flows through ``percept/clock.py`` (harness-only). The
doc was corrected with inline implementation-status notes.

These tests pin BOTH sides: the code facts that made the claims false, and the presence of the notes
that record them. If someone later wires ``percept/sequencing.py`` or ``percept/clock.py`` into the
production path, the code assertions here fail — which is the signal to go delete the corresponding
note from the ADR, not to weaken the test.
"""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path

import pytest

import percept
from percept import clock as clock_mod
from percept import gate as gate_mod

PACKAGE_ROOT = Path(percept.__file__).resolve().parent
ADR_PATH = PACKAGE_ROOT.parents[2] / "docs" / "adr" / "ADR-001-judgment-and-gate-contract.md"


@pytest.fixture(scope="module")
def adr_text() -> str:
    if not ADR_PATH.exists():  # docs are not shipped inside the wheel
        pytest.skip(f"ADR-001 not present at {ADR_PATH} (running outside the source tree)")
    return ADR_PATH.read_text(encoding="utf-8")


def _module_imports(path: Path) -> set[str]:
    """Every dotted name imported by ``path``, normalized so relative imports keep their leaf."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def _importers_of(module_leaf: str) -> set[Path]:
    """Source files under ``percept/`` that actually import ``module_leaf`` (any relative depth)."""
    hits: set[Path] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name == f"{module_leaf}.py":
            continue
        for imported in _module_imports(path):
            if imported.split(".")[-1] == module_leaf:
                hits.add(path.relative_to(PACKAGE_ROOT))
                break
    return hits


# --------------------------------------------------------------------------------------- §1


def test_gate_returns_gateoutcome_not_gatedecision() -> None:
    """The gate's shipped output type is GateOutcome; gate.py owns no GateDecision."""
    assert hasattr(gate_mod, "GateOutcome")
    assert not hasattr(gate_mod, "GateDecision")
    assert percept.GateOutcome is gate_mod.GateOutcome


def test_three_unrelated_gatedecision_types_still_exist() -> None:
    """The naming collision that forced 'GateOutcome' is real — two OTHER GateDecision types."""
    from percept.harness.tier0 import GateDecision as Tier0GateDecision
    from percept.sequencing import GateDecision as SequencingGateDecision

    assert Tier0GateDecision is not SequencingGateDecision
    # ...and neither is the gate's own outcome type.
    assert gate_mod.GateOutcome not in (Tier0GateDecision, SequencingGateDecision)


def test_adr_section_1_names_the_shipped_symbol(adr_text: str) -> None:
    assert "GateOutcome" in adr_text
    assert "percept.gate.GateOutcome" in adr_text


# --------------------------------------------------------------------------------------- §8


def test_sequencing_primitives_have_zero_production_call_sites() -> None:
    """EvidenceSequencer / SerializedDecisionReducer are constructed nowhere under percept/."""
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name == "sequencing.py":
            continue
        text = path.read_text(encoding="utf-8")
        code = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith(("#", "*", ":"))
        )
        if "EvidenceSequencer" in code or "SerializedDecisionReducer" in code:
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert offenders == [], (
        "percept/sequencing.py gained a production call site: "
        f"{offenders}. ADR-001 §8's implementation-status note is now stale — update the ADR."
    )


def test_only_harness_metrics_imports_the_sequencing_module() -> None:
    assert _importers_of("sequencing") == {Path("harness/metrics.py")}


def test_worker_scheduler_is_single_flight_with_one_latest_pending() -> None:
    """The mechanism that actually sidesteps the §8 race, so §8 is not needed at run time."""
    from percept.runtime.worker import _JudgeScheduler

    sched = _JudgeScheduler.__doc__ or ""
    assert "single-flight" in sched
    assert "one latest-pending" in sched
    src = (PACKAGE_ROOT / "runtime" / "worker.py").read_text(encoding="utf-8")
    assert "judge_superseded" in src


def test_adr_section_8_declares_itself_unimplemented(adr_text: str) -> None:
    section = adr_text.split("## §8")[1].split("## §9")[0]
    assert "zero production importers" in section
    assert "sidestep" in section.lower()
    assert "single-flight" in section


# --------------------------------------------------------------------------------------- §5


def test_clock_module_is_harness_only() -> None:
    importers = _importers_of("clock")
    assert importers, "percept/clock.py has no importers at all — the seam vanished"
    non_harness = {p for p in importers if p.parts[0] != "harness"}
    assert non_harness == set(), (
        f"percept/clock.py is now imported outside harness/: {sorted(map(str, non_harness))}. "
        "ADR-001 §5's implementation-status note is now stale — update the ADR."
    )


def test_production_time_defaults_are_bare_time_monotonic() -> None:
    """agent/gate/worker default their injectable clock to time.monotonic, not clock.WallClock."""
    for rel in ("agent.py", "runtime/worker.py"):
        src = (PACKAGE_ROOT / rel).read_text(encoding="utf-8")
        assert "time.monotonic" in src, rel
        assert "WallClock" not in src, rel


def test_wallclock_is_behaviourally_time_monotonic() -> None:
    """The gap is naming only — the note says 'behaviourally equivalent', so prove it."""
    assert clock_mod.WallClock is time.monotonic


def test_adr_section_5_states_the_seam_is_harness_only(adr_text: str) -> None:
    section = adr_text.split("## §5")[1].split("## §6")[0]
    assert "harness-only" in section
    assert "time.monotonic" in section


# --------------------------------------------------------------- Status line


def test_status_line_does_not_read_as_shipped(adr_text: str) -> None:
    status = next(line for line in adr_text.splitlines() if line.startswith("- **Status:**"))
    assert not re.fullmatch(r"- \*\*Status:\*\*\s*accepted\.?", status.strip())
    assert "partially implemented" in status.lower()
