"""The integration examples run offline as a developer would run them: no keys, no network."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"


def run(script: str, *args: str, cwd: Path = EXAMPLES) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    done = subprocess.run([sys.executable, str(EXAMPLES / script), *args], capture_output=True, text=True,
                          timeout=60, env=env, cwd=cwd)
    assert done.returncode == 0, done.stderr
    return done


def decisions(stdout: str) -> list[str]:
    return [line.split()[0] for line in stdout.splitlines() if re.match(r"^(met|cleared|uncertain|expired)\s", line)]


def test_kitchen_watch_reports_met_then_expired_then_cleared_and_acks_each():
    out = run("kitchen_watch.py").stdout
    assert decisions(out) == ["met", "expired", "cleared"], out
    assert "cause=max_open" in out and "cause=no" in out
    assert re.search(r"delivered 3, acked 3$", out.strip()), out


def test_frigate_semantic_reports_one_package_for_the_person_who_left_it():
    out = run("frigate_semantic.py").stdout
    assert decisions(out) == ["met"], out
    assert "(person b2)" in out and "a1" not in out and "c7" not in out


def test_frigate_semantic_replays_a_file_of_frigate_payloads(tmp_path):
    def person(oid: str, camera: str, frame_time: float) -> dict:
        return {"type": "new", "after": {"id": oid, "camera": camera, "label": "person", "frame_time": frame_time}}

    # Replayed relative to the first event, which plays 0.8 s in; the package appears at 2.0 s.
    events = [person("w0", "front_door", 1_700_000_000.0), person("x9", "front_door", 1_700_000_001.5),
              person("y1", "back_yard", 1_700_000_001.6)]
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    out = run("frigate_semantic.py", "--replay", str(path), cwd=tmp_path).stdout
    assert decisions(out) == ["met"], out
    assert "(person x9)" in out and "w0" not in out and "y1" not in out


def test_webhook_agent_posts_every_decision_and_acks_after_a_2xx():
    # About 7 s: the example plays a 7 s drawn clip at a demo request budget.
    out = run("webhook_agent.py", "--offline").stdout
    assert sorted(decisions(out)) == ["cleared", "met", "met"], out
    assert "met      person_fell" in out and "cleared  door_open" in out, out
    assert "webhook received 3 decisions" in out, out


def test_watch_plan_compiles_explains_and_runs():
    out = run("watch_plan.py").stdout
    assert "feasibility: \"best_effort\"" in out and "{'shelf_sensor': ('package_fell',)}" in out, out
    assert decisions(out) == ["met", "cleared"], out
