"""Tests for the deployment-owned judge catalog (ADR-009, PERCEPT-SDK-P06-001).

Covers: from_file happy path (incl. the supports_static_context default), fail-loud on
missing/extra keys and wrong types, malformed JSON, env-unset → empty catalog, env-set → loaded,
and spec-level validation (empty strings rejected).
"""

from __future__ import annotations

import json

import pytest

from percept.deployment_judges import (
    EMPTY_DEPLOYMENT_JUDGES,
    DeploymentJudges,
    DeploymentJudgesError,
    JudgeDeploymentSpec,
    load_deployment_judges,
)


def _write(path, obj) -> str:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def _catalog(**overrides) -> dict:
    base = {
        "available_judges": {
            "audit_specialist_judge@0.1": {
                "factory_ref": "claude",
                "provider_ref": "anthropic",
                "model_ref": "claude-x",
                "supports_static_context": True,
            }
        },
        "context_store_root": "/srv/judge-context",
    }
    base.update(overrides)
    return base


# --- from_file happy path ---------------------------------------------------


def test_from_file_happy_path(tmp_path):
    p = _write(tmp_path / "cat.json", _catalog())
    cat = DeploymentJudges.from_file(p)

    assert cat.authorizes("audit_specialist_judge@0.1")
    assert not cat.authorizes("unknown_judge@9.9")
    assert cat.context_store_root == "/srv/judge-context"

    spec = cat.spec_for("audit_specialist_judge@0.1")
    assert isinstance(spec, JudgeDeploymentSpec)
    assert spec.factory_ref == "claude"
    assert spec.provider_ref == "anthropic"
    assert spec.model_ref == "claude-x"
    assert spec.supports_static_context is True


def test_supports_static_context_defaults_true(tmp_path):
    obj = _catalog()
    del obj["available_judges"]["audit_specialist_judge@0.1"]["supports_static_context"]
    p = _write(tmp_path / "cat.json", obj)

    spec = DeploymentJudges.from_file(p).spec_for("audit_specialist_judge@0.1")
    assert spec.supports_static_context is True


def test_context_store_root_optional(tmp_path):
    obj = _catalog()
    del obj["context_store_root"]
    p = _write(tmp_path / "cat.json", obj)
    cat = DeploymentJudges.from_file(p)
    assert cat.context_store_root is None


def test_ref_is_opaque_no_version_parsing(tmp_path):
    # Two refs that share a stem are unrelated; only an exact-key match authorizes.
    obj = {
        "available_judges": {
            "audit_specialist_judge@0.1": {
                "factory_ref": "claude",
                "provider_ref": "anthropic",
                "model_ref": "m",
            }
        }
    }
    p = _write(tmp_path / "cat.json", obj)
    cat = DeploymentJudges.from_file(p)
    assert cat.authorizes("audit_specialist_judge@0.1")
    assert not cat.authorizes("audit_specialist_judge@0.2")
    assert not cat.authorizes("audit_specialist_judge")


# --- fail-loud: malformed JSON ----------------------------------------------


def test_malformed_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(str(p))


def test_missing_file_raises(tmp_path):
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(str(tmp_path / "nope.json"))


def test_non_object_root_raises(tmp_path):
    p = _write(tmp_path / "cat.json", [1, 2, 3])
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(p)


# --- fail-loud: missing / extra keys ----------------------------------------


def test_missing_available_judges_raises(tmp_path):
    p = _write(tmp_path / "cat.json", {"context_store_root": "/x"})
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(p)


def test_extra_catalog_key_raises(tmp_path):
    obj = _catalog()
    obj["surprise"] = 1
    p = _write(tmp_path / "cat.json", obj)
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(p)


def test_missing_spec_key_raises(tmp_path):
    obj = {
        "available_judges": {
            "j@0.1": {"factory_ref": "claude", "provider_ref": "anthropic"}
        }
    }
    p = _write(tmp_path / "cat.json", obj)
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(p)


def test_extra_spec_key_raises(tmp_path):
    obj = _catalog()
    obj["available_judges"]["audit_specialist_judge@0.1"]["rogue"] = "x"
    p = _write(tmp_path / "cat.json", obj)
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(p)


# --- fail-loud: wrong types --------------------------------------------------


def test_available_judges_wrong_type_raises(tmp_path):
    p = _write(tmp_path / "cat.json", {"available_judges": ["a"]})
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(p)


def test_spec_entry_wrong_type_raises(tmp_path):
    p = _write(tmp_path / "cat.json", {"available_judges": {"j@0.1": "notadict"}})
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(p)


def test_context_store_root_wrong_type_raises(tmp_path):
    obj = _catalog(context_store_root=123)
    p = _write(tmp_path / "cat.json", obj)
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(p)


def test_supports_static_context_wrong_type_raises(tmp_path):
    obj = _catalog()
    obj["available_judges"]["audit_specialist_judge@0.1"]["supports_static_context"] = "yes"
    p = _write(tmp_path / "cat.json", obj)
    with pytest.raises(DeploymentJudgesError):
        DeploymentJudges.from_file(p)


# --- spec validation ---------------------------------------------------------


@pytest.mark.parametrize("field_name", ["factory_ref", "provider_ref", "model_ref"])
def test_spec_empty_string_rejected(field_name):
    kwargs = {"factory_ref": "a", "provider_ref": "b", "model_ref": "c"}
    kwargs[field_name] = ""
    with pytest.raises(DeploymentJudgesError):
        JudgeDeploymentSpec(**kwargs)


@pytest.mark.parametrize("field_name", ["factory_ref", "provider_ref", "model_ref"])
def test_spec_whitespace_only_rejected(field_name):
    kwargs = {"factory_ref": "a", "provider_ref": "b", "model_ref": "c"}
    kwargs[field_name] = "   "
    with pytest.raises(DeploymentJudgesError):
        JudgeDeploymentSpec(**kwargs)


def test_spec_non_string_rejected():
    with pytest.raises(DeploymentJudgesError):
        JudgeDeploymentSpec(factory_ref="a", provider_ref="b", model_ref=5)  # type: ignore[arg-type]


# --- env resolution ----------------------------------------------------------


def test_load_env_unset_is_empty():
    cat = load_deployment_judges(env={})
    assert cat is EMPTY_DEPLOYMENT_JUDGES
    assert not cat.authorizes("anything@1.0")


def test_load_env_empty_string_is_empty():
    cat = load_deployment_judges(env={"PERCEPT_JUDGE_CATALOG": ""})
    assert cat is EMPTY_DEPLOYMENT_JUDGES


def test_load_env_set_loads_catalog(tmp_path):
    p = _write(tmp_path / "cat.json", _catalog())
    cat = load_deployment_judges(env={"PERCEPT_JUDGE_CATALOG": p})
    assert cat.authorizes("audit_specialist_judge@0.1")


def test_load_env_set_bad_path_raises():
    with pytest.raises(DeploymentJudgesError):
        load_deployment_judges(env={"PERCEPT_JUDGE_CATALOG": "/no/such/catalog.json"})


# --- immutability ------------------------------------------------------------


def test_empty_catalog_authorizes_nothing():
    assert not EMPTY_DEPLOYMENT_JUDGES.authorizes("x@1")
    assert EMPTY_DEPLOYMENT_JUDGES.spec_for("x@1") is None


def test_available_judges_is_read_only(tmp_path):
    cat = DeploymentJudges.from_file(_write(tmp_path / "cat.json", _catalog()))
    with pytest.raises(TypeError):
        cat.available_judges["hacked@1"] = None  # type: ignore[index]
