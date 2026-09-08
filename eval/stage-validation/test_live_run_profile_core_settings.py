from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import live_run as LR  # noqa: E402


def test_situation_profile_core_knobs_are_bound_before_watch_apply(monkeypatch):
    monkeypatch.setattr(
        LR,
        "situation_for",
        lambda _case: ("dynamic", "fixed", "transition"),
    )

    settings = LR.settings_for_case(LR.load_settings(), {"dataset": "offline-control"})

    assert settings.edge.salience_stat == "block_max"
    assert settings.edge.motion_gate_block_max == 0.20
    assert settings.edge.ego_compensate is False
    assert settings.gate.require_false_first is True
    assert settings.gate.refractory_s == 15.0
