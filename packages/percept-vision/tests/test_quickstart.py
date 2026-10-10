"""The offline quickstart runs as a new developer would run it: no keys, no network, no OpenCV."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

QUICKSTART = Path(__file__).resolve().parents[3] / "examples" / "quickstart.py"


def test_the_quickstart_prints_met_cleared_uncertain_coverage_and_a_matching_replay():
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    done = subprocess.run([sys.executable, str(QUICKSTART)], capture_output=True, text=True, timeout=30,
                          env=env, cwd=QUICKSTART.parent)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    types = [line.split()[0] for line in lines if re.match(r"^(met|cleared|uncertain|expired)\s", line)]
    assert types == ["met", "cleared", "uncertain"], done.stdout
    assert any(line.startswith("coverage:") for line in lines)
    assert re.fullmatch(r"replay: 3 decisions from \d+ inputs reproduced exactly", lines[-1]), lines[-1]
    assert any(re.match(r"^  box_in_zone\s+looked at \d+ times", line) for line in lines), done.stdout
    assert "cv2" not in done.stderr
