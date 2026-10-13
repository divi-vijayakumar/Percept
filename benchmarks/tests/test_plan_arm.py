"""The percept-plan-<name> arm picks one plan per case by its source id and compiles it."""
import json
from types import SimpleNamespace

import pytest

from benchmarks import run as R
from percept.judges import FakeJudge


def plan(source):
    return {"schema_version": "percept.watch/v1alpha1", "plan_id": f"p-{source}", "revision": 1,
            "sources": [{"id": source, "kind": "video"}], "budget": {"max_requests_per_min": 24},
            "concerns": [{"id": "enter", "source": source, "description": "a person enters",
                          "judgment": {"route": "judge"}, "decision": {"kind": "event", "rearm_s": 30.0}}]}


def test_the_plan_whose_source_is_in_the_case_id_is_compiled(tmp_path):
    (tmp_path / "zone").mkdir()
    for cam in ("G341", "G423"):
        (tmp_path / "zone" / f"{cam}.json").write_text(json.dumps(plan(cam)))
    args = SimpleNamespace(percept_plans=str(tmp_path))
    case = SimpleNamespace(case_id="2018-03-05.13-15-00.13-20-00.hospital.G341")
    c = R.plan_for(case, "percept-plan-zone", args, FakeJudge({}))
    assert c.monitor_kwargs["max_requests_per_min"] == 24 and [x.id for x in c.conditions] == ["enter"]
    assert c.conditions[0].rearm_s == 30.0
    with pytest.raises(ValueError, match="0 plans match"):
        R.plan_for(SimpleNamespace(case_id="x.G999"), "percept-plan-zone", args, FakeJudge({}))
    cfg = R.monitor_config("percept-plan-zone", SimpleNamespace(percept_plans=str(tmp_path)), FakeJudge({}))
    assert set(cfg["plans"]) == {"G341.json", "G423.json"} and cfg["plans"]["G341.json"]["max_requests_per_min"] == 24
