from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SDK = ROOT / "packages" / "percept-vision"
TOOL = ROOT / "tools" / "percept_wheel_smoke.py"


def test_source_mode_exercises_public_multimodal_product_paths():
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--allow-source"],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(SDK),
            "PERCEPT_NO_BANNER": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["schema_version"] == "percept.wheel_smoke/1"
    assert report["product_paths"]["video"]["incidents"] == 1
    assert report["product_paths"]["video"]["asks"] == 1
    assert report["product_paths"]["audio"]["incidents"] == 1
    assert report["product_paths"]["fused"]["incidents"] == 1
    assert report["product_paths"]["deadline"]["outcomes"] == [
        "deadline",
        "cancelled",
    ]
    assert report["product_paths"]["packs"] == [
        "physical_scene_starter@1/package_appeared",
        "ego_motion_starter@1/scene_transition",
    ]


def test_qualification_mode_refuses_the_source_checkout():
    completed = subprocess.run(
        [sys.executable, str(TOOL)],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(SDK),
            "PERCEPT_NO_BANNER": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "percept resolved from source tree" in completed.stderr


def test_artifact_name_audit_rejects_internal_paths_and_top_level_eval():
    namespace = runpy.run_path(
        str(TOOL),
        run_name="wheel_smoke_artifact_audit_test",
    )
    audit = namespace["_audit_artifact_names"]
    for names, match in (
        (["percept/x.py", "scr" + "atch/launch-plan.md"], "forbidden internal"),
        (["percept/x.py", "eval/bench.py"], "unexpected top-level"),
    ):
        try:
            audit(names, wheel=True)
        except AssertionError as exc:
            assert match in str(exc)
        else:
            raise AssertionError("artifact audit accepted a forbidden path")
