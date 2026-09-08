"""Regression guard for the live evaluator's SDK deadline ownership."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def test_unset_eval_deadlines_follow_the_sdk_defaults() -> None:
    env = dict(os.environ)
    env.pop("PERCEPT_SV_PROVIDER_DEADLINE_S", None)
    env.pop("PERCEPT_SV_DECISION_DEADLINE_S", None)
    env["PYTHONPATH"] = str(REPO / "packages" / "percept-vision")
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(HERE)!r}); "
        "import live_run as lr; "
        "from percept.backends.judge_pool import "
        "REFLEX_PROVIDER_DEADLINE_S, REFLEX_DECISION_DEADLINE_S; "
        "assert lr.ACTION_PROVIDER_DEADLINE_S == REFLEX_PROVIDER_DEADLINE_S; "
        "assert lr.ACTION_DECISION_DEADLINE_S == REFLEX_DECISION_DEADLINE_S"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_explicit_eval_deadline_arm_remains_available() -> None:
    env = dict(os.environ)
    env["PERCEPT_SV_PROVIDER_DEADLINE_S"] = "1.5"
    env["PERCEPT_SV_DECISION_DEADLINE_S"] = "1.9"
    env["PYTHONPATH"] = str(REPO / "packages" / "percept-vision")
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(HERE)!r}); "
        "import live_run as lr; "
        "assert lr.ACTION_PROVIDER_DEADLINE_S == 1.5; "
        "assert lr.ACTION_DECISION_DEADLINE_S == 1.9"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
