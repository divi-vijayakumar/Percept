"""Card C2 — exit-criteria proof: SourceProfile wired into the runtime via the two-step binder
(``percept.resolver.ConfigResolver`` -> ``percept.runtime_compiler.runtime_compile``).

The failure this card exists to kill: a screen-watch deployment silently admits nothing until
someone hand-exports ``PERCEPT_SALIENCE_STAT=block_max`` + ``PERCEPT_MOTION_GATE_BLOCK_MAX=0.20``.
Four bars, each pinned by a test class below:

1. The localized screen fixture admits ~11/12 with an EXPLICIT ``screen_document`` profile and
   ZERO env vars — the profile alone must be enough, mirroring ``run_doctor`` usage.
2. Precedence, both directions: env beats a profile default; an explicit WatchPlan field beats env.
3. Env overriding an EXPLICITLY selected profile emits the loud ``env_overrides_explicit_profile``
   diagnostic — proven end-to-end through ``ConfigResolver`` -> ``runtime_compile``.
4. A still-camera, motion-gated ``physical_scene`` watch admitting 0 stays non-alarming
   byte-for-byte (the regression this card must never break) — through the same compile path.
"""

from __future__ import annotations

import os

import pytest

from percept.config import Settings
from percept.contracts.watchplan import WatchPlan
from percept.doctor import ZERO_ADMITS_ON_FIXTURE, run_doctor
from percept.harness.tier0 import Frame
from percept.resolver import ConfigResolver
from percept.runtime_compiler import runtime_compile
from percept.source_profiles_default import PHYSICAL_SCENE, SCREEN_DOCUMENT

W, H = 48, 36
GRID = (8, 6)


def _blank() -> Frame:
    return Frame(w=W, h=H, pixels=tuple([0] * (W * H)))


def _bright_blocks(n: int, value: int = 255) -> Frame:
    px = [0] * (W * H)
    bw, bh = W // GRID[0], H // GRID[1]
    for b in range(n):
        bx = (b % GRID[0]) * bw
        by = (b // GRID[0]) * bh
        for y in range(by, by + bh):
            for x in range(bx, bx + bw):
                px[y * W + x] = value
    return Frame(w=W, h=H, pixels=tuple(px))


def _localized_stream(n: int = 12, blocks: int = 4) -> list[tuple[float, Frame]]:
    """The MEASURED BASELINE fixture (mirrors tests/test_doctor.py::_localized_stream): alternating
    blank / four-lit-blocks — block_max clears its cutoff, the whole-frame mean does not."""
    lit = _bright_blocks(blocks)
    return [(float(i), _blank() if i % 2 == 0 else lit) for i in range(n)]


def _quiet_stream(n: int = 12) -> list[tuple[float, Frame]]:
    return [(float(i), _blank()) for i in range(n)]


def _clear_percept_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("PERCEPT_"):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# 1. baseline fixture: 0/12 with nothing declared -> ~11/12 with an EXPLICIT
#    screen_document profile and ZERO env vars
# ---------------------------------------------------------------------------


class TestZeroEnvExplicitProfileAdmits:
    def test_no_profile_no_env_admits_0_of_12(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The alarm this card exists to kill: nothing declared, zero admits, silently."""
        _clear_percept_env(monkeypatch)
        report = run_doctor(frames=_localized_stream(n=12, blocks=4), goals=["a cell changes"])
        assert report.goals[0].admitted == 0
        assert report.has(ZERO_ADMITS_ON_FIXTURE)

    def test_explicit_screen_document_profile_zero_env_admits_about_11_of_12(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fix: route the SAME fixture through ConfigResolver -> runtime_compile with an
        EXPLICIT screen_document profile (tuned @1: block_max@0.20) and ZERO env vars, then feed the
        compiled Settings into run_doctor exactly the way a real caller's Settings would arrive —
        mirroring run_doctor usage, not hand-setting Settings.edge fields."""
        _clear_percept_env(monkeypatch)
        resolver = ConfigResolver()
        plan = WatchPlan(condition="a cell changes")
        resolved = resolver.resolve(plan, source_profile=SCREEN_DOCUMENT, source_profile_explicit=True)
        compiled = runtime_compile(resolved)

        # Prove zero env vars were read for the WINNING value: the profile's own declaration.
        assert resolved.provenance["Settings.edge.salience_stat"].owner_layer == "SourceProfile"
        assert compiled.settings.edge.salience_stat == "block_max"
        assert compiled.settings.edge.motion_gate_block_max == pytest.approx(0.20)

        report = run_doctor(
            frames=_localized_stream(n=12, blocks=4), goals=["a cell changes"],
            settings=compiled.settings,
        )
        assert report.goals[0].active_stat == "block_max"
        assert report.goals[0].admitted >= 11
        assert not report.has(ZERO_ADMITS_ON_FIXTURE)


# ---------------------------------------------------------------------------
# 2 & 3. precedence, both directions, plus the loud diagnostic — through the compile path
# ---------------------------------------------------------------------------


class TestPrecedenceThroughTheCompilePath:
    def test_env_beats_a_non_explicit_profile_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A RESOLVED-DEFAULT profile (source_profile_explicit=False) is outranked by env — no
        diagnostic, because nobody explicitly asked for this profile's value."""
        _clear_percept_env(monkeypatch)
        monkeypatch.setenv("PERCEPT_SALIENCE_STAT", "p95")
        resolver = ConfigResolver()
        resolved = resolver.resolve(
            WatchPlan(condition="x"), source_profile=SCREEN_DOCUMENT, source_profile_explicit=False,
        )
        compiled = runtime_compile(resolved)
        assert compiled.settings.edge.salience_stat == "p95"
        assert resolved.provenance["Settings.edge.salience_stat"].owner_layer == "Deployment"
        assert resolver.last_receipt.status == "applied"
        assert resolver.last_receipt.warnings == ()

    def test_explicit_watchplan_field_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The highest tier in the chain: an explicit WatchPlan field outranks env, whatever env says."""
        _clear_percept_env(monkeypatch)
        monkeypatch.setenv("PERCEPT_SALIENCE_STAT", "p95")
        resolver = ConfigResolver()
        plan = WatchPlan(condition="x", explicit={"WatchProfile.salience_stat": "mean"})
        resolved = resolver.resolve(plan, source_profile=SCREEN_DOCUMENT, source_profile_explicit=True)
        compiled = runtime_compile(resolved)
        assert compiled.settings.edge.salience_stat == "mean"
        assert resolved.provenance["Settings.edge.salience_stat"].owner_layer == "WatchPlan"

    def test_env_overriding_an_explicit_profile_emits_the_loud_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The resolver's existing machinery, proven end-to-end: env overriding a field owned by an
        EXPLICITLY selected profile is loud (a WatchReceipt warning), not silent — precisely the
        "supply PERCEPT_SALIENCE_STAT by hand and hope" failure mode, now announced instead of hidden."""
        _clear_percept_env(monkeypatch)
        monkeypatch.setenv("PERCEPT_SALIENCE_STAT", "p95")
        resolver = ConfigResolver()
        resolved = resolver.resolve(
            WatchPlan(condition="x"), source_profile=SCREEN_DOCUMENT, source_profile_explicit=True,
        )
        compiled = runtime_compile(resolved)
        assert compiled.settings.edge.salience_stat == "p95"  # env still wins the VALUE

        receipt = resolver.last_receipt
        assert receipt is not None
        assert receipt.status == "applied_with_warnings"
        codes = [w.code for w in receipt.warnings]
        assert "env_overrides_explicit_profile" in codes
        warning = next(w for w in receipt.warnings if w.code == "env_overrides_explicit_profile")
        assert warning.path == "Settings.edge.salience_stat"
        assert "block_max" in warning.message  # names the profile's overridden value
        assert "p95" in warning.message         # and the env value that won


# ---------------------------------------------------------------------------
# 4. regression: a still, motion-gated physical_scene watch admitting 0 stays non-alarming,
#    byte-for-byte, through the same compile path this card wires.
# ---------------------------------------------------------------------------


class TestStillCameraRegressionThroughTheCompilePath:
    def test_still_camera_physical_scene_zero_admit_stays_non_alarming(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_percept_env(monkeypatch)
        resolver = ConfigResolver()
        plan = WatchPlan(condition="someone walks in")
        resolved = resolver.resolve(plan, source_profile=PHYSICAL_SCENE, source_profile_explicit=True)
        compiled = runtime_compile(resolved)

        # physical_scene's own tuned default (block_max@0.20) is wired, same as pre-C2 env-set behavior.
        assert compiled.settings.edge.salience_stat == "block_max"

        report = run_doctor(
            frames=_quiet_stream(12), goals=["someone walks in"], settings=compiled.settings,
        )
        assert report.goals[0].admitted == 0
        finding = next(f for f in report.findings if f.code == ZERO_ADMITS_ON_FIXTURE)
        # Byte-for-byte the pre-existing framing: a WARN, never an error, never a runtime alarm.
        assert finding.severity == "warn"
        assert "not automatically a failure" in finding.detail

    def test_still_camera_no_coverage_floor_declared_by_physical_scene(self) -> None:
        """PHYSICAL_SCENE declares no coverage_floor -- runtime_compile's coverage_floor stays None,
        so percept.runtime.worker.run's zero_admission_coverage classification is untouched
        (see tests/test_runtime.py / tests/test_tracing_model.py for the full degraded-classification
        proof this regression depends on)."""
        resolver = ConfigResolver()
        resolved = resolver.resolve(
            WatchPlan(condition="someone walks in"), source_profile=PHYSICAL_SCENE,
        )
        compiled = runtime_compile(resolved)
        assert compiled.coverage_floor is None
