from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "tools" / "qualify_hosted_byok_native.py"
spec = importlib.util.spec_from_file_location("native_gate", MODULE)
assert spec and spec.loader
native_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_gate)


def test_native_gate_is_not_executed_without_authenticated_result(tmp_path: Path) -> None:
    recipe = tmp_path / "Dockerfile"
    recipe.write_text("FROM exact\n")
    evidence = native_gate.canonical_evidence(source_sha="a" * 40, source_tree="b" * 40, recipe=recipe, result=None)
    assert evidence["status"] == "not_executed"
    assert evidence["platform"] == "linux/amd64"
    assert evidence["base_digest"] == native_gate.BASE


def test_native_gate_requires_every_exact_containment_proof(tmp_path: Path) -> None:
    recipe = tmp_path / "Dockerfile"
    recipe.write_text("FROM exact\n")
    result = {"schema": 1, "status": "passed", "cases": list(native_gate.CASES), "environment_poisoned": True, "no_descendants": True, "joined": True, "late_effect": False}
    assert native_gate.canonical_evidence(source_sha="a" * 40, source_tree="b" * 40, recipe=recipe, result=result)["status"] == "passed"
    result["late_effect"] = True
    try:
        native_gate.canonical_evidence(source_sha="a" * 40, source_tree="b" * 40, recipe=recipe, result=result)
    except ValueError:
        return
    raise AssertionError("late effects must fail closed")


def test_native_runner_extracts_only_reviewed_stdlib_helpers() -> None:
    runner = Path(__file__).parents[1] / "tools" / "hosted_byok_native_matrix.py"
    text = runner.read_text()
    assert "import pytest" not in text
    assert "_run_production_composition_harness" in text
    assert "linux" in text and "x86_64" in text


def test_recipe_pins_base_platform_and_source_identity_labels() -> None:
    recipe = (Path(__file__).parents[1] / "Dockerfile.hosted-byok-native").read_text()
    assert "python:3.12-slim@sha256:cab2dbf575e971934a81e4622f5aba17aa7929719bd7e31033a3a83b97fd0464" in recipe
    assert "ARG SOURCE_SHA" in recipe and "ARG SOURCE_TREE" in recipe
    assert "ENTRYPOINT [\"/usr/local/bin/python\", \"-I\"" in recipe
