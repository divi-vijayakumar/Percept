from __future__ import annotations

from types import SimpleNamespace

import live_run


def test_authored_goal_state_detectors_project_to_typed_watchplan_field() -> None:
    authored = SimpleNamespace(
        explicit={
            "Goal.state.detectors": ["fixed-camera-collision-candidate"],
            "Goal.state.cadence.gate_on": "semantic",
        }
    )

    plan = live_run.watch_plan_for("g", "vehicles collide", authored)

    assert plan.detectors == ("fixed-camera-collision-candidate",)
    assert "Goal.state.detectors" not in plan.explicit
    assert plan.explicit["Goal.state.cadence.gate_on"] == "semantic"
